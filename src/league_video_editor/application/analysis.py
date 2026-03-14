"""Application-facing multi-signal analysis workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..analyze import (
    _collect_local_ai_cues,
    _collect_ocr_cues,
    _compute_vision_window_scores,
)
from ..config.settings import DetectionConfig, ProjectConfig
from ..core.audio_analyzer import extract_audio_rms_samples
from ..core.highlight_detector import HighlightEvent, SignalWeights, run_highlight_detection
from ..core.parallel import PipelineStage, parallel_stages
from ..core.smart_editor import EditPlan, generate_smart_edit_plan
from ..ffmpeg_utils import (
    detect_scene_events_adaptive,
    extract_signalstats_samples,
    probe_duration_seconds,
)
from ..models import TranscriptionCue, VisionWindow
from .content import build_content_package
from .overlays import build_overlay_bundle


_DEFAULT_WHISPER_CUE_THRESHOLD = 0.22
_DEFAULT_OCR_CUE_THRESHOLD = 0.18


@dataclass
class AnalysisBundle:
    duration_seconds: float
    scene_threshold_used: float
    scene_events: list[float]
    vision_windows: list[VisionWindow]
    whisper_cues: list[TranscriptionCue]
    ocr_cues: list[TranscriptionCue]
    audio_rms_samples: list[tuple[float, float]]
    highlights: list[HighlightEvent]
    edit_plan: EditPlan
    content_package: dict[str, Any]
    overlay_bundle: dict[str, Any]


def analyze_project_video(
    *,
    input_path: Path,
    project_config: ProjectConfig,
) -> AnalysisBundle:
    """Run the full product analysis workflow for a single project."""

    duration_seconds = probe_duration_seconds(input_path)
    scene_events, scene_threshold_used = detect_scene_events_adaptive(
        input_path,
        project_config.detection.scene_threshold,
        duration_seconds=duration_seconds,
        show_progress=False,
    )

    vision_windows = _extract_vision_windows(
        input_path=input_path,
        scene_events=scene_events,
        duration_seconds=duration_seconds,
        sample_fps=1.0,
    )

    enrichment = _collect_enrichment_signals(
        input_path=input_path,
        duration_seconds=duration_seconds,
        detection=project_config.detection,
    )
    whisper_cues = enrichment.get("whisper_cues") or []
    ocr_cues = enrichment.get("ocr_cues") or []
    audio_rms_samples = enrichment.get("audio_rms_samples") or []

    highlights = run_highlight_detection(
        scene_events=scene_events,
        vision_windows=vision_windows,
        whisper_cues=whisper_cues,
        ocr_cues=ocr_cues,
        audio_rms_samples=audio_rms_samples,
        weights=_resolve_signal_weights(project_config.detection),
        max_highlights=project_config.editing.max_clips,
        duration_seconds=duration_seconds,
    )
    edit_plan = generate_smart_edit_plan(
        highlights,
        source_duration=duration_seconds,
        target_duration=project_config.editing.target_duration_seconds,
        min_clip_seconds=project_config.editing.min_clip_seconds,
        max_clip_seconds=project_config.editing.max_clip_seconds,
        pre_fight_context=project_config.editing.pre_fight_context_seconds,
        post_fight_aftermath=project_config.editing.post_fight_aftermath_seconds,
        dynamic_length=project_config.editing.dynamic_length_enabled,
        retain_death_context=project_config.editing.retain_death_context,
        min_gap_between_clips=project_config.editing.min_gap_between_clips,
        max_clips=project_config.editing.max_clips,
    )

    content_package = build_content_package(
        champion=project_config.champion,
        content=project_config.content,
        output=project_config.output,
        upload=project_config.upload,
        highlights=[serialize_highlight(item) for item in highlights],
        edit_plan=edit_plan.to_dict(),
        match_duration_seconds=duration_seconds,
    )
    project_config.content.thumbnail_headline = content_package.get("thumbnail_headline", "")
    overlay_bundle = build_overlay_bundle(
        champion=project_config.champion,
        content=project_config.content,
        overlay_config=project_config.overlays,
        upload=project_config.upload,
        output_dir=Path(project_config.output_dir),
    )

    return AnalysisBundle(
        duration_seconds=duration_seconds,
        scene_threshold_used=scene_threshold_used,
        scene_events=scene_events,
        vision_windows=vision_windows,
        whisper_cues=whisper_cues,
        ocr_cues=ocr_cues,
        audio_rms_samples=audio_rms_samples,
        highlights=highlights,
        edit_plan=edit_plan,
        content_package=content_package,
        overlay_bundle=overlay_bundle,
    )


def serialize_analysis_bundle(bundle: AnalysisBundle) -> dict[str, Any]:
    return {
        "duration_seconds": bundle.duration_seconds,
        "scene_threshold_used": bundle.scene_threshold_used,
        "scene_events": bundle.scene_events,
        "vision_windows": [serialize_vision_window(item) for item in bundle.vision_windows],
        "whisper_cues": [serialize_cue(item) for item in bundle.whisper_cues],
        "ocr_cues": [serialize_cue(item) for item in bundle.ocr_cues],
        "highlights": [serialize_highlight(item) for item in bundle.highlights],
        "edit_plan": bundle.edit_plan.to_dict(),
        "content_package": bundle.content_package,
        "overlay_bundle": bundle.overlay_bundle,
    }


def serialize_highlight(item: HighlightEvent) -> dict[str, Any]:
    return {
        "timestamp": round(item.timestamp, 3),
        "start": round(item.start, 3),
        "end": round(item.end, 3),
        "duration": round(item.duration, 3),
        "score": round(item.score, 4),
        "event_type": item.event_type,
        "label": item.label,
        "signals": {key: round(value, 4) for key, value in item.signals.items()},
        "metadata": item.metadata,
    }


def serialize_vision_window(item: VisionWindow) -> dict[str, Any]:
    return {
        "start": round(item.start, 3),
        "end": round(item.end, 3),
        "score": round(item.score, 4),
        "motion": round(item.motion, 4),
        "saturation": round(item.saturation, 4),
        "scene_density": round(item.scene_density, 4),
    }


def serialize_cue(item: TranscriptionCue) -> dict[str, Any]:
    return {
        "start": round(item.start, 3),
        "end": round(item.end, 3),
        "score": round(item.score, 4),
        "text": item.text,
        "keywords": list(item.keywords),
    }


def extract_vision_windows(
    *,
    input_path: Path,
    scene_events: list[float],
    duration_seconds: float,
    sample_fps: float,
) -> list[VisionWindow]:
    return _extract_vision_windows(
        input_path=input_path,
        scene_events=scene_events,
        duration_seconds=duration_seconds,
        sample_fps=sample_fps,
    )


def collect_enrichment_signals(
    *,
    input_path: Path,
    duration_seconds: float,
    detection: DetectionConfig,
) -> dict[str, Any]:
    return _collect_enrichment_signals(
        input_path=input_path,
        duration_seconds=duration_seconds,
        detection=detection,
    )


def resolve_signal_weights(config: DetectionConfig) -> SignalWeights:
    return _resolve_signal_weights(config)


def _extract_vision_windows(
    *,
    input_path: Path,
    scene_events: list[float],
    duration_seconds: float,
    sample_fps: float,
) -> list[VisionWindow]:
    samples = extract_signalstats_samples(
        input_path,
        duration_seconds=duration_seconds,
        sample_fps=sample_fps,
        show_progress=False,
    )
    return _compute_vision_window_scores(
        frame_samples=samples,
        events=scene_events,
        duration_seconds=duration_seconds,
    )


def _collect_enrichment_signals(
    *,
    input_path: Path,
    duration_seconds: float,
    detection: DetectionConfig,
) -> dict[str, Any]:
    stages: list[PipelineStage] = []

    whisper_model = _resolve_default_whisper_model()
    if detection.whisper_enabled and whisper_model is not None:
        stages.append(
            PipelineStage(
                name="whisper_cues",
                weight=0.4,
                func=_collect_local_ai_cues,
                kwargs={
                    "input_path": input_path,
                    "whisper_model": whisper_model,
                    "whisper_binary": "auto",
                    "whisper_language": "en",
                    "whisper_threads": 4,
                    "cue_threshold": _DEFAULT_WHISPER_CUE_THRESHOLD,
                },
            )
        )

    if (
        detection.killfeed_enabled
        or detection.objective_detection_enabled
        or detection.scoreboard_detection_enabled
    ):
        stages.append(
            PipelineStage(
                name="ocr_cues",
                weight=0.35,
                func=_collect_ocr_cues,
                kwargs={
                    "input_path": input_path,
                    "tesseract_binary": "auto",
                    "sample_fps": 0.25,
                    "cue_threshold": _DEFAULT_OCR_CUE_THRESHOLD,
                },
            )
        )

    if detection.audio_excitement_enabled:
        stages.append(
            PipelineStage(
                name="audio_rms_samples",
                weight=0.25,
                func=extract_audio_rms_samples,
                kwargs={
                    "input_path": input_path,
                    "duration_seconds": duration_seconds,
                    "show_progress": False,
                },
            )
        )

    return parallel_stages(stages)


def _resolve_signal_weights(config: DetectionConfig) -> SignalWeights:
    return SignalWeights(
        scene_change=config.scene_weight,
        kill_feed=config.killfeed_weight if config.killfeed_enabled else 0.0,
        combat_intensity=config.combat_intensity_weight if config.combat_intensity_enabled else 0.0,
        objectives=config.objective_weight if config.objective_detection_enabled else 0.0,
        scoreboard_spike=config.scoreboard_weight if config.scoreboard_detection_enabled else 0.0,
        transcript=config.whisper_weight if config.whisper_enabled else 0.0,
        audio_excitement=config.audio_excitement_weight if config.audio_excitement_enabled else 0.0,
    )


def _resolve_default_whisper_model() -> Path | None:
    candidates: list[Path] = []
    for root in (Path.cwd(), Path.cwd() / "models", Path(__file__).resolve().parents[3] / "models"):
        if root.is_file() and root.suffix == ".bin":
            candidates.append(root)
        elif root.exists():
            candidates.extend(sorted(root.glob("ggml-*.bin")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
