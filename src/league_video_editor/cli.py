"""Thin CLI entry point.

All heavy logic lives in the sub-modules:
  cache.py, models.py, ffmpeg_utils.py, analyze.py,
  render.py, thumbnail.py, description.py, watchability.py, upload.py

This file keeps argparse wiring, validate helpers, and backward-compatible
re-exports so that existing tests importing from league_video_editor.cli
continue to work without changes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Re-exports from new modules (keeps existing tests working)
# ---------------------------------------------------------------------------

from .analyze import (
    _boost_vision_windows_with_ai_cues,
    _collect_local_ai_cues,
    _collect_ocr_cues,
    _compute_vision_window_scores,
    _detect_combat_cues,
    _detect_death_cues,
    _detect_gameplay_start,
    _parse_whisper_json_cues,
    _rank_vision_candidates,
    analyze_recording,
    build_auto_optimize_variants as _build_auto_optimize_variants,
    build_segments as _build_segments_impl,
    merge_segments,
    read_plan,
    sample_evenly,
    score_plan_analytically,
    score_transcript_text as _score_transcript_text,
    score_vision_activity,
    select_optimize_metric_value as _select_optimize_metric_value,
    write_plan,
)
from .cache import (
    CacheStore,
    DEFAULT_CACHE_DIR,
    DEFAULT_CACHE_MAX_GB,
    DEFAULT_CACHE_TTL_DAYS,
)
from .description import (
    _clean_moment_blurb,
    generate_youtube_description as _generate_youtube_description_impl,
)
from .ffmpeg_utils import (
    build_filter_complex as _build_filter_complex,
    detect_crop_filter as _detect_crop_filter_impl,
    detect_scene_events,
    detect_scene_events_adaptive as _detect_scene_events_adaptive_impl,
    estimate_render_duration as _estimate_render_duration,
    extract_audio_for_whisper as _extract_audio_for_whisper_impl,
    extract_loudnorm_json as _extract_loudnorm_json,
    extract_ocr_frames as _extract_ocr_frames,
    has_audio_stream as _has_audio_stream,
    looks_like_loudnorm_nonfinite_error as _looks_like_loudnorm_nonfinite_error,
    probe_duration_seconds as _probe_duration_seconds_impl,
    render_progress_line,
    require_binary,
    resolve_encoder,
    resolve_whisper_cpp_binary as _resolve_whisper_cpp_binary,
    resolve_tesseract_binary as _resolve_tesseract_binary,
    _run_command_with_stderr_tail,
    run_command as _run_command,
    run_tesseract_ocr as _run_tesseract_ocr,
    video_codec_args as _video_codec_args,
    video_postprocess_filter as _video_postprocess_filter,
)
from .upload import (
    DEFAULT_CLIENT_SECRETS,
    DEFAULT_TOKEN_PATH,
    upload_to_youtube,
)
from .models import (
    ADAPTIVE_AI_CUE_THRESHOLD_FACTORS,
    ADAPTIVE_AI_CUE_THRESHOLD_MIN,
    ADAPTIVE_SCENE_THRESHOLD_FACTORS,
    ADAPTIVE_SCENE_THRESHOLD_MIN,
    DEFAULT_AI_CUE_THRESHOLD,
    DEFAULT_AUTO_OPTIMIZE_CANDIDATES,
    DEFAULT_AUTO_OPTIMIZE_METRIC,
    DEFAULT_DESCRIPTION_MAX_MOMENTS,
    DEFAULT_DESCRIPTION_OUTPUT,
    DEFAULT_DESCRIPTION_TITLE_COUNT,
    DEFAULT_INTRO_SECONDS,
    DEFAULT_OCR_CUE_SCORING,
    DEFAULT_OCR_CUE_THRESHOLD,
    DEFAULT_OCR_SAMPLE_FPS,
    DEFAULT_ONE_SHOT_SMART,
    DEFAULT_OUTRO_SECONDS,
    DEFAULT_PROFILE,
    DEFAULT_RESULT_DETECT_FPS,
    DEFAULT_RESULT_DETECT_TAIL_SECONDS,
    DEFAULT_TARGET_DURATION_RATIO,
    DEFAULT_TARGET_DURATION_SECONDS,
    DEFAULT_TESSERACT_BIN,
    DEFAULT_THUMBNAIL_CHAMPION_ANCHOR,
    DEFAULT_THUMBNAIL_CHAMPION_SCALE,
    DEFAULT_THUMBNAIL_HEIGHT,
    DEFAULT_THUMBNAIL_HEADLINE_COLOR,
    DEFAULT_THUMBNAIL_HEADLINE_SIZE,
    DEFAULT_THUMBNAIL_HEADLINE_Y_RATIO,
    DEFAULT_THUMBNAIL_QUALITY,
    DEFAULT_THUMBNAIL_SCENE_THRESHOLD,
    DEFAULT_THUMBNAIL_VISION_SAMPLE_FPS,
    DEFAULT_THUMBNAIL_VISION_STEP_SECONDS,
    DEFAULT_THUMBNAIL_VISION_WINDOW_SECONDS,
    DEFAULT_THUMBNAIL_WIDTH,
    DEFAULT_VISION_SAMPLE_FPS,
    DEFAULT_VISION_SCORING,
    DEFAULT_VISION_STEP_SECONDS,
    DEFAULT_VISION_WINDOW_SECONDS,
    DEFAULT_WATCHABILITY_SCENE_THRESHOLD,
    DEFAULT_WHISPER_CPP_BIN,
    DEFAULT_WHISPER_LANGUAGE,
    DEFAULT_WHISPER_THREADS,
    DEFAULT_WHISPER_VAD,
    DEFAULT_WHISPER_VAD_THRESHOLD,
    ENCODER_PROFILES,
    EditorError,
    Segment,
    TranscriptionCue,
    VIDEO_ENCODERS,
    VisionWindow,
)
from .render import (
    render_highlights as _render_highlights_impl,
    transcode_full_match as _transcode_full_match_impl,
)
from .thumbnail import (
    _create_headline_overlay_image,
    generate_thumbnail as _generate_thumbnail_impl,
)
from .watchability import _build_watchability_report


# ---------------------------------------------------------------------------
# CLI-layer wrappers — defined here so tests can patch cli-local names
# ---------------------------------------------------------------------------
# These wrappers pass cli-local function references (which unittest.mock can
# patch via league_video_editor.cli._xxx) into the underlying implementations.


def _probe_duration_seconds(input_path: Path) -> float:
    """CLI-layer wrapper so patches on ``cli._run_command`` propagate."""
    return _probe_duration_seconds_impl(input_path, _run_cmd_fn=_run_command)


def detect_scene_events_adaptive(
    input_path: Path,
    scene_threshold: float,
    *,
    duration_seconds: float | None = None,
    show_progress: bool = False,
    progress_label: str | None = "Analyzing scenes",
    progress_callback=None,
    minimum_threshold: float = 0.08,
    factors: tuple = (1.0, 0.75, 0.6, 0.45),
) -> tuple:
    """CLI-layer wrapper so patches on ``cli.detect_scene_events`` propagate."""
    return _detect_scene_events_adaptive_impl(
        input_path,
        scene_threshold,
        duration_seconds=duration_seconds,
        show_progress=show_progress,
        progress_label=progress_label,
        progress_callback=progress_callback,
        minimum_threshold=minimum_threshold,
        factors=factors,
        _detect_fn=detect_scene_events,
    )


def _detect_watchability_crop_filter(
    input_path: Path,
    *,
    duration_seconds: float,
) -> "str | None":
    """CLI-layer wrapper so patches on ``cli._run_command`` propagate."""
    return _detect_crop_filter_impl(
        input_path,
        duration_seconds=duration_seconds,
        _run_cmd_fn=_run_command,
    )


def generate_thumbnail(
    *,
    input_path: Path,
    output_path: Path,
    timestamp_seconds: "float | None",
    scene_threshold: float,
    vision_sample_fps: float,
    vision_window_seconds: float,
    vision_step_seconds: float,
    width: int,
    height: int,
    quality: int,
    auto_crop: bool = True,
    enhance: bool = True,
    champion_overlay_path: "Path | None" = None,
    champion_scale: float,
    champion_anchor: str,
    headline_text: "str | None" = None,
    headline_size: int,
    headline_color: str,
    headline_font: "Path | None" = None,
    headline_y_ratio: float,
) -> "dict[str, float | int | bool]":
    """CLI-layer wrapper so patches on cli-local names propagate."""
    return _generate_thumbnail_impl(
        input_path=input_path,
        output_path=output_path,
        timestamp_seconds=timestamp_seconds,
        scene_threshold=scene_threshold,
        vision_sample_fps=vision_sample_fps,
        vision_window_seconds=vision_window_seconds,
        vision_step_seconds=vision_step_seconds,
        width=width,
        height=height,
        quality=quality,
        auto_crop=auto_crop,
        enhance=enhance,
        champion_overlay_path=champion_overlay_path,
        champion_scale=champion_scale,
        champion_anchor=champion_anchor,
        headline_text=headline_text,
        headline_size=headline_size,
        headline_color=headline_color,
        headline_font=headline_font,
        headline_y_ratio=headline_y_ratio,
        _probe_fn=_probe_duration_seconds,
        _detect_crop_fn=_detect_watchability_crop_filter,
        _create_headline_fn=_create_headline_overlay_image,
        _run_cmd_fn=_run_command,
        _detect_events_fn=detect_scene_events_adaptive,
        _score_vision_fn=score_vision_activity,
    )


def generate_youtube_description(
    *,
    input_path: Path,
    output_path: Path,
    champion: str,
    channel_name: str,
    scene_threshold: float,
    vision_sample_fps: float,
    vision_window_seconds: float,
    vision_step_seconds: float,
    whisper_model: "Path | None" = None,
    whisper_bin: str,
    whisper_language: str,
    whisper_threads: int,
    whisper_audio_stream: int = -1,
    whisper_vad: bool = True,
    whisper_vad_threshold: float = 0.50,
    whisper_vad_model: "Path | None" = None,
    ocr_cue_scoring: str = "auto",
    tesseract_bin: str = "auto",
    ocr_sample_fps: float = 0.25,
    ocr_cue_threshold: float = 0.16,
    ai_cue_threshold: float = 0.40,
    max_moments: int = 8,
    title_count: int = 3,
) -> "dict[str, object]":
    """CLI-layer wrapper so patches on cli-local names propagate."""
    return _generate_youtube_description_impl(
        input_path=input_path,
        output_path=output_path,
        champion=champion,
        channel_name=channel_name,
        scene_threshold=scene_threshold,
        vision_sample_fps=vision_sample_fps,
        vision_window_seconds=vision_window_seconds,
        vision_step_seconds=vision_step_seconds,
        whisper_model=whisper_model,
        whisper_bin=whisper_bin,
        whisper_language=whisper_language,
        whisper_threads=whisper_threads,
        whisper_audio_stream=whisper_audio_stream,
        whisper_vad=whisper_vad,
        whisper_vad_threshold=whisper_vad_threshold,
        whisper_vad_model=whisper_vad_model,
        ocr_cue_scoring=ocr_cue_scoring,
        tesseract_bin=tesseract_bin,
        ocr_sample_fps=ocr_sample_fps,
        ocr_cue_threshold=ocr_cue_threshold,
        ai_cue_threshold=ai_cue_threshold,
        max_moments=max_moments,
        title_count=title_count,
        _probe_fn=_probe_duration_seconds,
        _detect_events_fn=detect_scene_events_adaptive,
        _score_vision_fn=score_vision_activity,
        _collect_ai_fn=_collect_local_ai_cues,
        _collect_ocr_fn=_collect_ocr_cues,
    )


def _run_with_loudnorm_fallback(
    command_with_loudnorm: list,
    command_without_loudnorm: "list | None",
    *,
    progress_label: "str | None" = None,
    progress_duration_seconds: "float | None" = None,
) -> None:
    """CLI-layer wrapper so patches on ``cli._run_command_with_stderr_tail`` propagate."""
    from .ffmpeg_utils import run_with_loudnorm_fallback as _impl
    _impl(
        command_with_loudnorm,
        command_without_loudnorm,
        progress_label=progress_label,
        progress_duration_seconds=progress_duration_seconds,
        _run_stderr_fn=_run_command_with_stderr_tail,
    )


def _extract_audio_for_whisper(
    *,
    input_path: Path,
    output_audio_path: Path,
    audio_stream_index: "int | None" = None,
) -> None:
    """CLI-layer wrapper so patches on ``cli._run_command`` propagate."""
    _extract_audio_for_whisper_impl(
        input_path=input_path,
        output_audio_path=output_audio_path,
        audio_stream_index=audio_stream_index,
        _run_cmd_fn=_run_command,
    )


def render_highlights(
    *,
    input_path: Path,
    plan_path: Path,
    output_path: Path,
    crf: int,
    preset: str,
    video_encoder: str,
    allow_upscale: bool,
    auto_crop: bool = False,
    crossfade_seconds: float,
    audio_fade_seconds: float,
    two_pass_loudnorm: bool,
) -> None:
    _render_highlights_impl(
        input_path=input_path,
        plan_path=plan_path,
        output_path=output_path,
        crf=crf,
        preset=preset,
        video_encoder=video_encoder,
        allow_upscale=allow_upscale,
        auto_crop=auto_crop,
        crossfade_seconds=crossfade_seconds,
        audio_fade_seconds=audio_fade_seconds,
        two_pass_loudnorm=two_pass_loudnorm,
        _read_plan_fn=read_plan,
        _has_audio_fn=_has_audio_stream,
        _run_cmd_fn=_run_command,
        _run_with_loudnorm_fn=_run_with_loudnorm_fallback,
    )


def transcode_full_match(
    *,
    input_path: Path,
    output_path: Path,
    crf: int,
    preset: str,
    video_encoder: str,
    allow_upscale: bool,
    auto_crop: bool = False,
    two_pass_loudnorm: bool,
) -> None:
    _transcode_full_match_impl(
        input_path=input_path,
        output_path=output_path,
        crf=crf,
        preset=preset,
        video_encoder=video_encoder,
        allow_upscale=allow_upscale,
        auto_crop=auto_crop,
        two_pass_loudnorm=two_pass_loudnorm,
        _has_audio_fn=_has_audio_stream,
        _run_cmd_fn=_run_command,
        _run_with_loudnorm_fn=_run_with_loudnorm_fallback,
    )


def _collect_local_ai_cues(
    *,
    input_path: Path,
    whisper_model: Path,
    whisper_binary: str,
    whisper_language: str,
    whisper_threads: int,
    cue_threshold: float,
    whisper_audio_stream: int | None = None,
    whisper_vad: bool = DEFAULT_WHISPER_VAD,
    whisper_vad_threshold: float = DEFAULT_WHISPER_VAD_THRESHOLD,
    whisper_vad_model: Path | None = None,
) -> list[TranscriptionCue]:
    from .analyze import _collect_local_ai_cues as _impl
    return _impl(
        input_path=input_path,
        whisper_model=whisper_model,
        whisper_binary=whisper_binary,
        whisper_language=whisper_language,
        whisper_threads=whisper_threads,
        cue_threshold=cue_threshold,
        whisper_audio_stream=whisper_audio_stream,
        whisper_vad=whisper_vad,
        whisper_vad_threshold=whisper_vad_threshold,
        whisper_vad_model=whisper_vad_model,
        _resolve_binary_fn=_resolve_whisper_cpp_binary,
        _extract_audio_fn=_extract_audio_for_whisper,
        _run_cmd_fn=_run_command,
    )


def _collect_ocr_cues(
    *,
    input_path: Path,
    tesseract_binary: str,
    sample_fps: float,
    cue_threshold: float,
) -> list[TranscriptionCue]:
    from .analyze import _collect_ocr_cues as _impl
    return _impl(
        input_path=input_path,
        tesseract_binary=tesseract_binary,
        sample_fps=sample_fps,
        cue_threshold=cue_threshold,
        _resolve_tesseract_fn=_resolve_tesseract_binary,
        _extract_frames_fn=_extract_ocr_frames,
        _run_ocr_fn=_run_tesseract_ocr,
    )


# ---------------------------------------------------------------------------
# analyze_watchability — defined here so tests can patch cli-local names
# (league_video_editor.cli._probe_duration_seconds, etc.)
# ---------------------------------------------------------------------------


def analyze_watchability(
    *,
    input_path: Path,
    scene_threshold: float = DEFAULT_WATCHABILITY_SCENE_THRESHOLD,
    vision_sample_fps: float = DEFAULT_VISION_SAMPLE_FPS,
    vision_window_seconds: float = DEFAULT_VISION_WINDOW_SECONDS,
    vision_step_seconds: float = DEFAULT_VISION_STEP_SECONDS,
    show_progress: bool = True,
) -> dict[str, object]:
    """Analyze a rendered video and return a watchability/quality report.

    Defined in cli.py (not re-exported from watchability.py) so that test
    patches on ``league_video_editor.cli._probe_duration_seconds``,
    ``league_video_editor.cli.detect_scene_events_adaptive``, and
    ``league_video_editor.cli.score_vision_activity`` work correctly.
    """
    from collections.abc import Callable

    last_progress: list[float] = [-1.0]

    def emit_progress(ratio: float) -> None:
        clamped = min(1.0, max(0.0, ratio))
        if clamped < last_progress[0] + 0.01 and clamped < 1.0:
            return
        line = render_progress_line("Analyzing watchability", clamped)
        if clamped >= 1.0:
            print(line, file=sys.stderr, flush=True)
        else:
            print(line, end="", file=sys.stderr, flush=True)
        last_progress[0] = clamped

    def stage_callback(start: float, end: float) -> Callable[[float], None]:
        span = max(0.0, end - start)

        def update(stage_ratio: float) -> None:
            emit_progress(start + span * min(1.0, max(0.0, stage_ratio)))

        return update

    if show_progress:
        emit_progress(0.0)

    duration_seconds = _probe_duration_seconds(input_path)
    if show_progress:
        emit_progress(0.05)

    events, scene_threshold_used = detect_scene_events_adaptive(
        input_path,
        scene_threshold,
        duration_seconds=duration_seconds,
        show_progress=show_progress,
        progress_label=None if show_progress else "Analyzing scenes",
        progress_callback=stage_callback(0.05, 0.55) if show_progress else None,
    )
    if show_progress:
        emit_progress(0.55)

    vision_windows = score_vision_activity(
        input_path,
        events=events,
        duration_seconds=duration_seconds,
        sample_fps=vision_sample_fps,
        window_seconds=vision_window_seconds,
        step_seconds=vision_step_seconds,
        show_progress=show_progress,
        progress_label=None if show_progress else "Scoring gameplay",
        progress_callback=stage_callback(0.55, 0.95) if show_progress else None,
    )
    if show_progress:
        emit_progress(0.95)

    report = _build_watchability_report(
        duration_seconds=duration_seconds,
        events=events,
        vision_windows=vision_windows,
        scene_threshold_used=scene_threshold_used,
    )
    if show_progress:
        emit_progress(1.0)
    return report


# ---------------------------------------------------------------------------
# Backward-compat 2-tuple wrapper for build_segments
# (tests unpack as: segments, used_fallback = build_segments(...))
# ---------------------------------------------------------------------------

def build_segments(
    events: list[float],
    *,
    duration_seconds: float,
    clip_before: float,
    clip_after: float,
    min_gap_seconds: float,
    max_clips: int,
    target_duration_seconds: float = DEFAULT_TARGET_DURATION_SECONDS,
    intro_seconds: float = DEFAULT_INTRO_SECONDS,
    outro_seconds: float = DEFAULT_OUTRO_SECONDS,
    vision_windows: list[VisionWindow] | None = None,
    ai_priority_cues: list[float] | None = None,
    ai_priority_details: list[TranscriptionCue] | None = None,
    force_outro_to_duration_end: bool = False,
    forced_cue_share: float = 0.60,
    ensure_game_phase_coverage: bool = True,
) -> tuple[list[Segment], bool]:
    """Backward-compatible 2-tuple variant that uses cli-local cue detector refs.

    Passing ``_death_cue_fn=_detect_death_cues`` and
    ``_combat_cue_fn=_detect_combat_cues`` makes unittest.mock patches on
    ``league_video_editor.cli._detect_death_cues`` and
    ``league_video_editor.cli._detect_combat_cues`` work correctly.
    """
    segments, used_fallback, _ = _build_segments_impl(
        events,
        duration_seconds=duration_seconds,
        clip_before=clip_before,
        clip_after=clip_after,
        min_gap_seconds=min_gap_seconds,
        max_clips=max_clips,
        target_duration_seconds=target_duration_seconds,
        intro_seconds=intro_seconds,
        outro_seconds=outro_seconds,
        vision_windows=vision_windows,
        ai_priority_cues=ai_priority_cues,
        ai_priority_details=ai_priority_details,
        force_outro_to_duration_end=force_outro_to_duration_end,
        forced_cue_share=forced_cue_share,
        ensure_game_phase_coverage=ensure_game_phase_coverage,
        _death_cue_fn=_detect_death_cues,
        _combat_cue_fn=_detect_combat_cues,
    )
    return segments, used_fallback


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_file(input_path: Path) -> None:
    if not input_path.exists():
        raise EditorError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise EditorError(f"Input path is not a file: {input_path}")


def _require_finite(option_name: str, value: float) -> None:
    if not math.isfinite(value):
        raise EditorError(f"--{option_name} must be a finite number.")


def _validate_cli_options(args: argparse.Namespace) -> None:  # noqa: C901
    if args.command in {"analyze", "auto"}:
        vision_sample_fps = float(getattr(args, "vision_sample_fps", DEFAULT_VISION_SAMPLE_FPS))
        vision_window_seconds = float(
            getattr(args, "vision_window_seconds", DEFAULT_VISION_WINDOW_SECONDS)
        )
        vision_step_seconds = float(
            getattr(args, "vision_step_seconds", DEFAULT_VISION_STEP_SECONDS)
        )
        vision_scoring = str(getattr(args, "vision_scoring", DEFAULT_VISION_SCORING))
        whisper_model = getattr(args, "whisper_model", None)
        whisper_bin = str(getattr(args, "whisper_bin", DEFAULT_WHISPER_CPP_BIN))
        whisper_language = str(getattr(args, "whisper_language", DEFAULT_WHISPER_LANGUAGE))
        whisper_threads = int(getattr(args, "whisper_threads", DEFAULT_WHISPER_THREADS))
        whisper_audio_stream = int(getattr(args, "whisper_audio_stream", -1))
        whisper_vad = bool(getattr(args, "whisper_vad", DEFAULT_WHISPER_VAD))
        whisper_vad_threshold = float(
            getattr(args, "whisper_vad_threshold", DEFAULT_WHISPER_VAD_THRESHOLD)
        )
        whisper_vad_model = getattr(args, "whisper_vad_model", None)
        ocr_cue_scoring = str(getattr(args, "ocr_cue_scoring", DEFAULT_OCR_CUE_SCORING))
        tesseract_bin = str(getattr(args, "tesseract_bin", DEFAULT_TESSERACT_BIN))
        ocr_sample_fps = float(getattr(args, "ocr_sample_fps", DEFAULT_OCR_SAMPLE_FPS))
        ocr_cue_threshold = float(getattr(args, "ocr_cue_threshold", DEFAULT_OCR_CUE_THRESHOLD))
        ai_cue_threshold = float(getattr(args, "ai_cue_threshold", DEFAULT_AI_CUE_THRESHOLD))
        end_on_result = bool(getattr(args, "end_on_result", True))
        result_detect_fps = float(getattr(args, "result_detect_fps", DEFAULT_RESULT_DETECT_FPS))
        result_detect_tail_seconds = float(
            getattr(args, "result_detect_tail_seconds", DEFAULT_RESULT_DETECT_TAIL_SECONDS)
        )
        target_duration_ratio = float(
            getattr(args, "target_duration_ratio", DEFAULT_TARGET_DURATION_RATIO)
        )
        _require_finite("scene-threshold", args.scene_threshold)
        if not 0.0 <= args.scene_threshold <= 1.0:
            raise EditorError("--scene-threshold must be between 0.0 and 1.0.")

        for opt, val in (
            ("clip-before", args.clip_before),
            ("clip-after", args.clip_after),
            ("min-gap-seconds", args.min_gap_seconds),
            ("target-duration-seconds", args.target_duration_seconds),
            ("intro-seconds", args.intro_seconds),
            ("outro-seconds", args.outro_seconds),
            ("vision-sample-fps", vision_sample_fps),
            ("vision-window-seconds", vision_window_seconds),
            ("vision-step-seconds", vision_step_seconds),
            ("whisper-vad-threshold", whisper_vad_threshold),
            ("ocr-sample-fps", ocr_sample_fps),
            ("ocr-cue-threshold", ocr_cue_threshold),
            ("ai-cue-threshold", ai_cue_threshold),
            ("result-detect-fps", result_detect_fps),
            ("result-detect-tail-seconds", result_detect_tail_seconds),
        ):
            _require_finite(opt, val)
            if val < 0:
                raise EditorError(f"--{opt} must be >= 0.")
        _require_finite("target-duration-ratio", target_duration_ratio)
        if not 0.0 <= target_duration_ratio <= 1.0:
            raise EditorError("--target-duration-ratio must be between 0.0 and 1.0.")
        if args.target_duration_seconds < 0:
            raise EditorError("--target-duration-seconds must be >= 0.")
        if vision_sample_fps <= 0:
            raise EditorError("--vision-sample-fps must be > 0.")
        if vision_window_seconds <= 0:
            raise EditorError("--vision-window-seconds must be > 0.")
        if vision_step_seconds <= 0:
            raise EditorError("--vision-step-seconds must be > 0.")
        if ocr_sample_fps <= 0:
            raise EditorError("--ocr-sample-fps must be > 0.")
        if end_on_result and result_detect_fps <= 0:
            raise EditorError("--result-detect-fps must be > 0 when --end-on-result is enabled.")
        if not 0.0 <= whisper_vad_threshold <= 1.0:
            raise EditorError("--whisper-vad-threshold must be between 0.0 and 1.0.")
        if not 0.0 <= ocr_cue_threshold <= 1.0:
            raise EditorError("--ocr-cue-threshold must be between 0.0 and 1.0.")
        if not 0.0 <= ai_cue_threshold <= 1.0:
            raise EditorError("--ai-cue-threshold must be between 0.0 and 1.0.")
        if vision_scoring not in {"off", "heuristic", "local-ai"}:
            raise EditorError("--vision-scoring must be 'off', 'heuristic', or 'local-ai'.")
        if ocr_cue_scoring not in {"off", "auto"}:
            raise EditorError("--ocr-cue-scoring must be 'off' or 'auto'.")
        if whisper_threads < 1:
            raise EditorError("--whisper-threads must be >= 1.")
        if whisper_audio_stream < -1:
            raise EditorError("--whisper-audio-stream must be >= -1.")
        if not whisper_bin.strip():
            raise EditorError("--whisper-bin must be a non-empty string.")
        if not whisper_language.strip():
            raise EditorError("--whisper-language must be a non-empty language code.")
        if whisper_vad_model is not None and not isinstance(whisper_vad_model, Path):
            raise EditorError("--whisper-vad-model must be a valid file path.")
        if not tesseract_bin.strip():
            raise EditorError("--tesseract-bin must be a non-empty string.")
        if vision_scoring == "local-ai" and whisper_model is None:
            raise EditorError("--whisper-model is required when --vision-scoring local-ai.")
        if args.max_clips < 1:
            raise EditorError("--max-clips must be >= 1.")
        if args.command == "auto":
            auto_optimize = bool(getattr(args, "auto_optimize", False))
            optimize_candidates = int(
                getattr(args, "optimize_candidates", DEFAULT_AUTO_OPTIMIZE_CANDIDATES)
            )
            optimize_metric = str(getattr(args, "optimize_metric", DEFAULT_AUTO_OPTIMIZE_METRIC))
            if optimize_candidates < 1:
                raise EditorError("--optimize-candidates must be >= 1.")
            if optimize_metric not in {"youtube", "watchability", "quality"}:
                raise EditorError(
                    "--optimize-metric must be 'youtube', 'watchability', or 'quality'."
                )
            if auto_optimize and optimize_candidates < 2:
                raise EditorError(
                    "--optimize-candidates must be >= 2 when --auto-optimize is enabled."
                )

    if args.command == "watchability":
        _require_finite("scene-threshold", args.scene_threshold)
        if not 0.0 <= args.scene_threshold <= 1.0:
            raise EditorError("--scene-threshold must be between 0.0 and 1.0.")
        for opt, val in (
            ("vision-sample-fps", args.vision_sample_fps),
            ("vision-window-seconds", args.vision_window_seconds),
            ("vision-step-seconds", args.vision_step_seconds),
        ):
            _require_finite(opt, val)
            if val <= 0:
                raise EditorError(f"--{opt} must be > 0.")

    if args.command == "thumbnail":
        _require_finite("scene-threshold", args.scene_threshold)
        if not 0.0 <= args.scene_threshold <= 1.0:
            raise EditorError("--scene-threshold must be between 0.0 and 1.0.")
        for opt, val in (
            ("vision-sample-fps", args.vision_sample_fps),
            ("vision-window-seconds", args.vision_window_seconds),
            ("vision-step-seconds", args.vision_step_seconds),
        ):
            _require_finite(opt, val)
            if val <= 0:
                raise EditorError(f"--{opt} must be > 0.")
        if args.timestamp is not None:
            _require_finite("timestamp", args.timestamp)
            if args.timestamp < 0:
                raise EditorError("--timestamp must be >= 0 when provided.")
        if args.width < 1:
            raise EditorError("--width must be >= 1.")
        if args.height < 1:
            raise EditorError("--height must be >= 1.")
        if not 2 <= args.quality <= 31:
            raise EditorError("--quality must be between 2 and 31.")
        _require_finite("champion-scale", args.champion_scale)
        if not 0.10 <= args.champion_scale <= 1.20:
            raise EditorError("--champion-scale must be between 0.10 and 1.20.")
        if args.champion_anchor not in {"left", "center", "right"}:
            raise EditorError("--champion-anchor must be one of: left, center, right.")
        if args.champion_overlay is not None:
            if not args.champion_overlay.exists() or not args.champion_overlay.is_file():
                raise EditorError(f"Champion overlay file not found: {args.champion_overlay}")
        if args.headline is not None and not args.headline.strip():
            raise EditorError("--headline must not be empty when provided.")
        if args.headline_size < 12:
            raise EditorError("--headline-size must be >= 12.")
        _require_finite("headline-y-ratio", args.headline_y_ratio)
        if not 0.0 <= args.headline_y_ratio <= 1.0:
            raise EditorError("--headline-y-ratio must be between 0.0 and 1.0.")
        if args.headline_font is not None:
            if not args.headline_font.exists() or not args.headline_font.is_file():
                raise EditorError(f"Headline font file not found: {args.headline_font}")
        if args.output.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise EditorError("--output must end with .jpg, .jpeg, or .png.")

    if args.command == "description":
        _require_finite("scene-threshold", args.scene_threshold)
        if not 0.0 <= args.scene_threshold <= 1.0:
            raise EditorError("--scene-threshold must be between 0.0 and 1.0.")
        for opt, val in (
            ("vision-sample-fps", args.vision_sample_fps),
            ("vision-window-seconds", args.vision_window_seconds),
            ("vision-step-seconds", args.vision_step_seconds),
            ("ocr-sample-fps", args.ocr_sample_fps),
            ("ocr-cue-threshold", args.ocr_cue_threshold),
            ("ai-cue-threshold", args.ai_cue_threshold),
            ("whisper-vad-threshold", args.whisper_vad_threshold),
        ):
            _require_finite(opt, val)
        if args.vision_sample_fps <= 0:
            raise EditorError("--vision-sample-fps must be > 0.")
        if args.vision_window_seconds <= 0:
            raise EditorError("--vision-window-seconds must be > 0.")
        if args.vision_step_seconds <= 0:
            raise EditorError("--vision-step-seconds must be > 0.")
        if args.ocr_sample_fps <= 0:
            raise EditorError("--ocr-sample-fps must be > 0.")
        if not 0.0 <= args.ocr_cue_threshold <= 1.0:
            raise EditorError("--ocr-cue-threshold must be between 0.0 and 1.0.")
        if not 0.0 <= args.ai_cue_threshold <= 1.0:
            raise EditorError("--ai-cue-threshold must be between 0.0 and 1.0.")
        if not 0.0 <= args.whisper_vad_threshold <= 1.0:
            raise EditorError("--whisper-vad-threshold must be between 0.0 and 1.0.")
        if args.whisper_threads < 1:
            raise EditorError("--whisper-threads must be >= 1.")
        if args.whisper_audio_stream < -1:
            raise EditorError("--whisper-audio-stream must be >= -1.")
        if args.ocr_cue_scoring not in {"off", "auto"}:
            raise EditorError("--ocr-cue-scoring must be 'off' or 'auto'.")
        if args.max_moments < 1:
            raise EditorError("--max-moments must be >= 1.")
        if args.title_count < 1:
            raise EditorError("--title-count must be >= 1.")
        if args.whisper_vad_model is not None and not isinstance(args.whisper_vad_model, Path):
            raise EditorError("--whisper-vad-model must be a valid file path.")
        if not str(args.whisper_bin).strip():
            raise EditorError("--whisper-bin must be a non-empty string.")
        if not str(args.whisper_language).strip():
            raise EditorError("--whisper-language must be a non-empty language code.")
        if not str(args.tesseract_bin).strip():
            raise EditorError("--tesseract-bin must be a non-empty string.")

    if args.command in {"render", "auto"}:
        for opt in ("crossfade-seconds", "audio-fade-seconds"):
            val = getattr(args, opt.replace("-", "_"))
            _require_finite(opt, val)
            if val < 0:
                raise EditorError(f"--{opt} must be >= 0.")

    if args.command in {"render", "auto", "full"}:
        if not 0 <= args.crf <= 51:
            raise EditorError("--crf must be between 0 and 51.")
        video_encoder = getattr(args, "video_encoder", "libx264")
        if video_encoder not in VIDEO_ENCODERS and video_encoder != "auto":
            raise EditorError(f"--video-encoder must be one of: {', '.join(VIDEO_ENCODERS)}.")
        if video_encoder not in {"libx264", "auto"} and args.preset != "medium":
            raise EditorError(
                "--preset applies only to libx264. Use the default preset with hardware encoders."
            )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _add_analysis_args(sub: argparse.ArgumentParser) -> None:
    """Add the shared analyze/auto analysis arguments."""
    sub.add_argument("--scene-threshold", type=float, default=0.35)
    sub.add_argument("--clip-before", type=float, default=12.0)
    sub.add_argument("--clip-after", type=float, default=18.0)
    sub.add_argument("--min-gap-seconds", type=float, default=20.0)
    sub.add_argument(
        "--max-clips", type=int, default=24,
        help="Maximum number of middle highlight clips between intro and outro anchors.",
    )
    sub.add_argument(
        "--target-duration-seconds", type=float, default=0.0,
        help="Absolute target duration override. 0 uses --target-duration-ratio.",
    )
    sub.add_argument(
        "--target-duration-ratio", type=float, default=DEFAULT_TARGET_DURATION_RATIO,
        help="Target highlights duration as a ratio of source match duration (default: 2/3).",
    )
    sub.add_argument(
        "--intro-seconds", type=float, default=DEFAULT_INTRO_SECONDS,
        help="Seconds to keep after detected in-game spawn/gameplay start.",
    )
    sub.add_argument(
        "--outro-seconds", type=float, default=DEFAULT_OUTRO_SECONDS,
        help="Seconds to keep at the end for match finish/nexus destruction.",
    )
    sub.add_argument(
        "--end-on-result", action=argparse.BooleanOptionalAction, default=True,
        help="Trim to detected Victory/Defeat end screen.",
    )
    sub.add_argument(
        "--result-detect-fps", type=float, default=DEFAULT_RESULT_DETECT_FPS,
        help="Frame sampling rate for end-of-match detection.",
    )
    sub.add_argument(
        "--result-detect-tail-seconds", type=float, default=DEFAULT_RESULT_DETECT_TAIL_SECONDS,
        help="How much of the recording tail to scan for end detection.",
    )
    sub.add_argument(
        "--one-shot-smart", action=argparse.BooleanOptionalAction, default=DEFAULT_ONE_SHOT_SMART,
        help="One-pass adaptive tuning to keep more kill context.",
    )
    sub.add_argument(
        "--vision-scoring", type=str, default=DEFAULT_VISION_SCORING,
        choices=["off", "heuristic", "local-ai"],
        help="Window scoring mode.",
    )
    sub.add_argument(
        "--vision-sample-fps", type=float, default=DEFAULT_VISION_SAMPLE_FPS,
        help="FPS for low-resolution frame sampling.",
    )
    sub.add_argument(
        "--vision-window-seconds", type=float, default=DEFAULT_VISION_WINDOW_SECONDS,
        help="Window size used to aggregate vision activity scores.",
    )
    sub.add_argument(
        "--vision-step-seconds", type=float, default=DEFAULT_VISION_STEP_SECONDS,
        help="Step size between consecutive vision scoring windows.",
    )
    sub.add_argument(
        "--whisper-model", type=Path, default=None,
        help="Path to a local whisper.cpp model file. Required for --vision-scoring local-ai.",
    )
    sub.add_argument(
        "--whisper-bin", type=str, default=DEFAULT_WHISPER_CPP_BIN,
        help="whisper.cpp executable name/path. Use 'auto' to search PATH.",
    )
    sub.add_argument(
        "--whisper-language", type=str, default=DEFAULT_WHISPER_LANGUAGE,
        help="Language code passed to whisper.cpp.",
    )
    sub.add_argument(
        "--whisper-threads", type=int, default=DEFAULT_WHISPER_THREADS,
        help="Thread count for whisper.cpp inference.",
    )
    sub.add_argument(
        "--whisper-audio-stream", type=int, default=-1,
        help="Audio stream index for whisper extraction (0-based). -1 for auto.",
    )
    sub.add_argument(
        "--whisper-vad", action=argparse.BooleanOptionalAction, default=DEFAULT_WHISPER_VAD,
        help="Enable whisper.cpp VAD segmentation.",
    )
    sub.add_argument(
        "--whisper-vad-threshold", type=float, default=DEFAULT_WHISPER_VAD_THRESHOLD,
        help="VAD threshold for whisper.cpp (0-1).",
    )
    sub.add_argument(
        "--whisper-vad-model", type=Path, default=None,
        help="Optional whisper.cpp VAD model path.",
    )
    sub.add_argument(
        "--ocr-cue-scoring", type=str, default=DEFAULT_OCR_CUE_SCORING, choices=["off", "auto"],
        help="OCR cue detection mode for on-screen kill/objective text.",
    )
    sub.add_argument(
        "--tesseract-bin", type=str, default=DEFAULT_TESSERACT_BIN,
        help="tesseract executable name/path. Use 'auto' to search PATH.",
    )
    sub.add_argument(
        "--ocr-sample-fps", type=float, default=DEFAULT_OCR_SAMPLE_FPS,
        help="Frame sampling rate for OCR cue extraction.",
    )
    sub.add_argument(
        "--ocr-cue-threshold", type=float, default=DEFAULT_OCR_CUE_THRESHOLD,
        help="OCR cue score threshold (0-1).",
    )
    sub.add_argument(
        "--ai-cue-threshold", type=float, default=DEFAULT_AI_CUE_THRESHOLD,
        help="Transcript cue score threshold (0-1).",
    )


def _add_render_args(sub: argparse.ArgumentParser, default_encoder: str = "libx264") -> None:
    """Add the shared render/auto/full encoding arguments."""
    sub.add_argument("--crf", type=int, default=20)
    sub.add_argument(
        "--video-encoder", type=str, default=default_encoder, choices=list(VIDEO_ENCODERS) + ["auto"],
        help="Video encoder. 'auto' picks VideoToolbox on Apple Silicon, libx264 elsewhere.",
    )
    sub.add_argument(
        "--allow-upscale", action="store_true",
        help="Upscale video up to 1080p.",
    )
    sub.add_argument(
        "--auto-crop", action=argparse.BooleanOptionalAction, default=True,
        help="Auto-detect and crop common black borders.",
    )
    sub.add_argument(
        "--two-pass-loudnorm", action="store_true",
        help="Use two-pass loudness normalization.",
    )
    sub.add_argument(
        "--preset", type=str, default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
        help="Encoder preset for libx264 only.",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lol-video-editor",
        description="Create YouTube-ready League of Legends videos from OBS recordings.",
    )

    # ── Global flags ──────────────────────────────────────────────────────
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        choices=["fast", "balanced", "quality"],
        help=(
            "Encoding + analysis profile. "
            "fast: h264_videotoolbox, lower sample FPS; "
            "balanced: auto encoder, standard FPS (default when flag omitted); "
            "quality: libx264 slow, two-pass loudnorm, full FPS."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Directory for persistent analysis cache (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the persistent analysis cache.",
    )
    parser.add_argument(
        "--cache-max-gb",
        type=float,
        default=DEFAULT_CACHE_MAX_GB,
        help=f"Maximum cache size in GB (default: {DEFAULT_CACHE_MAX_GB}).",
    )
    parser.add_argument(
        "--cache-ttl-days",
        type=float,
        default=DEFAULT_CACHE_TTL_DAYS,
        help=f"Cache entry TTL in days (default: {DEFAULT_CACHE_TTL_DAYS}).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── analyze ───────────────────────────────────────────────────────────
    analyze = subparsers.add_parser(
        "analyze", help="Detect highlight segments and write an edit plan JSON."
    )
    analyze.add_argument("input", type=Path, help="Path to OBS recording")
    analyze.add_argument("--plan", type=Path, default=Path("edit-plan.json"))
    _add_analysis_args(analyze)

    # ── render ────────────────────────────────────────────────────────────
    render = subparsers.add_parser(
        "render", help="Render highlights MP4 from an existing edit plan JSON."
    )
    render.add_argument("input", type=Path, help="Path to OBS recording")
    render.add_argument("--plan", type=Path, required=True, help="Path to edit plan JSON")
    render.add_argument("--output", type=Path, default=Path("highlights.mp4"))
    render.add_argument(
        "--crossfade-seconds", type=float, default=0.0,
        help="Crossfade duration between highlight clips.",
    )
    render.add_argument(
        "--audio-fade-seconds", type=float, default=0.03,
        help="Fade-in/out duration per clip.",
    )
    _add_render_args(render)

    # ── auto ──────────────────────────────────────────────────────────────
    auto = subparsers.add_parser(
        "auto", help="Analyze and render in one command (best default for first run).",
    )
    auto.add_argument("input", type=Path, help="Path to OBS recording")
    auto.add_argument("--plan", type=Path, default=Path("edit-plan.json"))
    auto.add_argument("--output", type=Path, default=Path("highlights.mp4"))
    _add_analysis_args(auto)
    auto.add_argument(
        "--crossfade-seconds", type=float, default=0.0,
        help="Crossfade duration between highlight clips.",
    )
    auto.add_argument(
        "--audio-fade-seconds", type=float, default=0.03,
        help="Fade-in/out duration per clip.",
    )
    _add_render_args(auto)
    auto.add_argument(
        "--auto-optimize", action=argparse.BooleanOptionalAction, default=False,
        help=(
            "Try multiple parameter variants and choose the best by --optimize-metric. "
            "Uses analytical scoring instead of rendering each candidate."
        ),
    )
    auto.add_argument(
        "--optimize-candidates", type=int, default=DEFAULT_AUTO_OPTIMIZE_CANDIDATES,
        help="Number of auto-optimize candidates to evaluate.",
    )
    auto.add_argument(
        "--optimize-metric", type=str, default=DEFAULT_AUTO_OPTIMIZE_METRIC,
        choices=["youtube", "watchability", "quality"],
        help="Metric used by auto-optimize candidate selection.",
    )

    # ── full ──────────────────────────────────────────────────────────────
    full = subparsers.add_parser(
        "full", help="Transcode full match to YouTube-ready format without clipping highlights.",
    )
    full.add_argument("input", type=Path, help="Path to OBS recording")
    full.add_argument("--output", type=Path, default=Path("full-match-youtube.mp4"))
    _add_render_args(full)

    # ── thumbnail ─────────────────────────────────────────────────────────
    thumbnail = subparsers.add_parser(
        "thumbnail", help="Generate a YouTube thumbnail frame from a video.",
    )
    thumbnail.add_argument("input", type=Path, help="Path to source video")
    thumbnail.add_argument("--output", type=Path, default=Path("thumbnail.jpg"))
    thumbnail.add_argument(
        "--timestamp", type=float, default=None,
        help="Optional timestamp in seconds. Auto-selects a high-action frame if omitted.",
    )
    thumbnail.add_argument(
        "--scene-threshold", type=float, default=DEFAULT_THUMBNAIL_SCENE_THRESHOLD,
    )
    thumbnail.add_argument(
        "--vision-sample-fps", type=float, default=DEFAULT_THUMBNAIL_VISION_SAMPLE_FPS,
    )
    thumbnail.add_argument(
        "--vision-window-seconds", type=float, default=DEFAULT_THUMBNAIL_VISION_WINDOW_SECONDS,
    )
    thumbnail.add_argument(
        "--vision-step-seconds", type=float, default=DEFAULT_THUMBNAIL_VISION_STEP_SECONDS,
    )
    thumbnail.add_argument("--width", type=int, default=DEFAULT_THUMBNAIL_WIDTH)
    thumbnail.add_argument("--height", type=int, default=DEFAULT_THUMBNAIL_HEIGHT)
    thumbnail.add_argument(
        "--quality", type=int, default=DEFAULT_THUMBNAIL_QUALITY,
        help="JPEG quality (2 best, 31 worst).",
    )
    thumbnail.add_argument(
        "--champion-overlay", type=Path, default=None,
        help="Path to transparent champion PNG to overlay on the thumbnail.",
    )
    thumbnail.add_argument(
        "--champion-scale", type=float, default=DEFAULT_THUMBNAIL_CHAMPION_SCALE,
    )
    thumbnail.add_argument(
        "--champion-anchor", type=str, default=DEFAULT_THUMBNAIL_CHAMPION_ANCHOR,
        choices=["left", "center", "right"],
    )
    thumbnail.add_argument("--headline", type=str, default=None)
    thumbnail.add_argument("--headline-size", type=int, default=DEFAULT_THUMBNAIL_HEADLINE_SIZE)
    thumbnail.add_argument("--headline-color", type=str, default=DEFAULT_THUMBNAIL_HEADLINE_COLOR)
    thumbnail.add_argument("--headline-font", type=Path, default=None)
    thumbnail.add_argument(
        "--headline-y-ratio", type=float, default=DEFAULT_THUMBNAIL_HEADLINE_Y_RATIO,
    )
    thumbnail.add_argument(
        "--auto-crop", action=argparse.BooleanOptionalAction, default=True,
    )
    thumbnail.add_argument(
        "--enhance", action=argparse.BooleanOptionalAction, default=True,
    )

    # ── description ───────────────────────────────────────────────────────
    description = subparsers.add_parser(
        "description",
        help="Generate CTR-focused YouTube title/description text from video events and cues.",
    )
    description.add_argument("input", type=Path, help="Path to source video")
    description.add_argument(
        "--output", type=Path, default=Path(DEFAULT_DESCRIPTION_OUTPUT),
    )
    description.add_argument("--report", type=Path, default=None)
    description.add_argument("--champion", type=str, default="")
    description.add_argument("--channel-name", type=str, default="")
    description.add_argument("--scene-threshold", type=float, default=0.20)
    description.add_argument(
        "--vision-sample-fps", type=float, default=DEFAULT_VISION_SAMPLE_FPS,
    )
    description.add_argument(
        "--vision-window-seconds", type=float, default=DEFAULT_VISION_WINDOW_SECONDS,
    )
    description.add_argument(
        "--vision-step-seconds", type=float, default=DEFAULT_VISION_STEP_SECONDS,
    )
    description.add_argument("--whisper-model", type=Path, default=None)
    description.add_argument("--whisper-bin", type=str, default=DEFAULT_WHISPER_CPP_BIN)
    description.add_argument("--whisper-language", type=str, default=DEFAULT_WHISPER_LANGUAGE)
    description.add_argument("--whisper-threads", type=int, default=DEFAULT_WHISPER_THREADS)
    description.add_argument("--whisper-audio-stream", type=int, default=-1)
    description.add_argument(
        "--whisper-vad", action=argparse.BooleanOptionalAction, default=DEFAULT_WHISPER_VAD,
    )
    description.add_argument(
        "--whisper-vad-threshold", type=float, default=DEFAULT_WHISPER_VAD_THRESHOLD,
    )
    description.add_argument("--whisper-vad-model", type=Path, default=None)
    description.add_argument(
        "--ocr-cue-scoring", type=str, default=DEFAULT_OCR_CUE_SCORING, choices=["off", "auto"],
    )
    description.add_argument("--tesseract-bin", type=str, default=DEFAULT_TESSERACT_BIN)
    description.add_argument(
        "--ocr-sample-fps", type=float, default=DEFAULT_OCR_SAMPLE_FPS,
    )
    description.add_argument(
        "--ocr-cue-threshold", type=float, default=DEFAULT_OCR_CUE_THRESHOLD,
    )
    description.add_argument(
        "--ai-cue-threshold", type=float, default=DEFAULT_AI_CUE_THRESHOLD,
    )
    description.add_argument(
        "--max-moments", type=int, default=DEFAULT_DESCRIPTION_MAX_MOMENTS,
    )
    description.add_argument(
        "--title-count", type=int, default=DEFAULT_DESCRIPTION_TITLE_COUNT,
    )

    # ── watchability ──────────────────────────────────────────────────────
    watchability = subparsers.add_parser(
        "watchability", help="Analyze a video and report a heuristic watchability score.",
    )
    watchability.add_argument("input", type=Path, help="Path to rendered video")
    watchability.add_argument(
        "--scene-threshold", type=float, default=DEFAULT_WATCHABILITY_SCENE_THRESHOLD,
    )
    watchability.add_argument(
        "--vision-sample-fps", type=float, default=DEFAULT_VISION_SAMPLE_FPS,
    )
    watchability.add_argument(
        "--vision-window-seconds", type=float, default=DEFAULT_VISION_WINDOW_SECONDS,
    )
    watchability.add_argument(
        "--vision-step-seconds", type=float, default=DEFAULT_VISION_STEP_SECONDS,
    )
    watchability.add_argument("--report", type=Path, default=None)

    # ── cache ─────────────────────────────────────────────────────────────
    cache_cmd = subparsers.add_parser(
        "cache", help="Manage the persistent analysis cache.",
    )
    cache_subs = cache_cmd.add_subparsers(dest="cache_action", required=True)
    cache_subs.add_parser("list", help="List all cached entries.")
    purge = cache_subs.add_parser("purge", help="Delete cached entries.")
    purge.add_argument(
        "--older-than-days", type=float, default=None,
        help="Only remove entries older than N days. Omit to remove all entries.",
    )
    cache_subs.add_parser("stats", help="Show cache size and entry count.")

    # ── upload ────────────────────────────────────────────────────────────
    upload = subparsers.add_parser(
        "upload", help="Upload a video to YouTube via the YouTube Data API v3.",
    )
    upload.add_argument("input", type=Path, help="Path to video file to upload")
    upload.add_argument("--title", type=str, required=True, help="Video title")
    upload.add_argument("--description", type=str, default="", help="Video description text")
    upload.add_argument(
        "--description-file", type=Path, default=None,
        help="Read description from a text file (overrides --description).",
    )
    upload.add_argument("--tags", type=str, nargs="*", default=None, help="Video tags")
    upload.add_argument(
        "--category-id", type=str, default="20",
        help="YouTube category ID (default: 20 = Gaming).",
    )
    upload.add_argument(
        "--privacy", type=str, default="private",
        choices=["private", "unlisted", "public"],
        help="Video privacy status (default: private).",
    )
    upload.add_argument("--thumbnail", type=Path, default=None, help="Custom thumbnail image")
    upload.add_argument(
        "--client-secrets", type=Path, default=DEFAULT_CLIENT_SECRETS,
        help=f"Path to OAuth client secrets JSON (default: {DEFAULT_CLIENT_SECRETS}).",
    )
    upload.add_argument(
        "--token-path", type=Path, default=DEFAULT_TOKEN_PATH,
        help=f"Path to save/load OAuth token (default: {DEFAULT_TOKEN_PATH}).",
    )

    # ── pipeline ──────────────────────────────────────────────────────────
    pipeline = subparsers.add_parser(
        "pipeline",
        help="Full pipeline: analyze → render → thumbnail → description → upload.",
    )
    pipeline.add_argument("input", type=Path, help="Path to OBS recording")
    pipeline.add_argument("--plan", type=Path, default=Path("edit-plan.json"))
    pipeline.add_argument("--output", type=Path, default=Path("highlights.mp4"))
    _add_analysis_args(pipeline)
    pipeline.add_argument(
        "--crossfade-seconds", type=float, default=0.0,
        help="Crossfade duration between highlight clips.",
    )
    pipeline.add_argument(
        "--audio-fade-seconds", type=float, default=0.03,
        help="Fade-in/out duration per clip.",
    )
    _add_render_args(pipeline)
    pipeline.add_argument(
        "--auto-optimize", action=argparse.BooleanOptionalAction, default=False,
    )
    pipeline.add_argument(
        "--optimize-candidates", type=int, default=DEFAULT_AUTO_OPTIMIZE_CANDIDATES,
    )
    pipeline.add_argument(
        "--optimize-metric", type=str, default=DEFAULT_AUTO_OPTIMIZE_METRIC,
        choices=["youtube", "watchability", "quality"],
    )
    # Thumbnail options
    pipeline.add_argument(
        "--thumbnail-output", type=Path, default=Path("thumbnail.jpg"),
    )
    pipeline.add_argument("--champion-overlay", type=Path, default=None)
    pipeline.add_argument("--headline", type=str, default=None)
    pipeline.add_argument("--headline-font", type=Path, default=None)
    # Description options
    pipeline.add_argument("--champion", type=str, default="")
    pipeline.add_argument("--channel-name", type=str, default="")
    pipeline.add_argument(
        "--description-output", type=Path, default=Path(DEFAULT_DESCRIPTION_OUTPUT),
    )
    # Upload options
    pipeline.add_argument(
        "--upload", action=argparse.BooleanOptionalAction, default=True,
        help="Upload to YouTube after rendering (default: yes).",
    )
    pipeline.add_argument(
        "--privacy", type=str, default="private",
        choices=["private", "unlisted", "public"],
        help="YouTube privacy status (default: private).",
    )
    pipeline.add_argument("--tags", type=str, nargs="*", default=None)
    pipeline.add_argument(
        "--client-secrets", type=Path, default=DEFAULT_CLIENT_SECRETS,
    )
    pipeline.add_argument(
        "--token-path", type=Path, default=DEFAULT_TOKEN_PATH,
    )

    return parser


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # ── Apply --profile defaults before validation ────────────────────────
    profile_name: str | None = getattr(args, "profile", None)
    if profile_name is not None:
        profile = ENCODER_PROFILES[profile_name]
        # Only override if user did not explicitly set a conflicting flag.
        # Since argparse defaults are set, we always apply profile values
        # for render-related settings on commands that have them.
        for attr, value in profile.items():
            if hasattr(args, attr):
                current = getattr(args, attr)
                # Detect if the value is still the argparse default
                default_val = parser._defaults.get(attr)  # type: ignore[attr-defined]
                if current == default_val or current is None:
                    setattr(args, attr, value)
        # video_encoder: resolve "auto" via encoder detection
        if hasattr(args, "video_encoder"):
            raw_encoder = getattr(args, "video_encoder", "libx264")
            if raw_encoder == "auto":
                setattr(args, "video_encoder", resolve_encoder("auto"))

    try:
        # ── Cache setup ───────────────────────────────────────────────────
        cache: CacheStore | None = None
        if args.command not in {"cache"} and not getattr(args, "no_cache", False):
            cache = CacheStore(
                cache_dir=args.cache_dir,
                max_gb=args.cache_max_gb,
                ttl_days=args.cache_ttl_days,
            )

        # ── cache subcommand ──────────────────────────────────────────────
        if args.command == "cache":
            store = CacheStore(
                cache_dir=args.cache_dir,
                max_gb=args.cache_max_gb,
                ttl_days=args.cache_ttl_days,
            )
            if args.cache_action == "stats":
                stats = store.stats()
                print(f"Cache directory: {stats['cache_dir']}")
                print(f"Entries: {stats['entry_count']}")
                print(f"Size: {stats['total_size_mb']:.1f} MB / {stats['max_gb']:.1f} GB limit")
                print(f"TTL: {stats['ttl_days']} days")
            elif args.cache_action == "list":
                entries = store.list_entries()
                if not entries:
                    print("Cache is empty.")
                else:
                    for entry in entries:
                        print(f"  {entry.key[:12]}… {entry.artifact:20s} {entry.size_bytes:8d}B")
                    print(f"Total: {len(entries)} entries")
            elif args.cache_action == "purge":
                removed = store.purge(
                    older_than_days=getattr(args, "older_than_days", None)
                )
                print(f"Removed {removed} cache entries.")
            return 0

        # ── upload (no ffmpeg needed) ────────────────────────────────────
        if args.command == "upload":
            desc_text = args.description
            if args.description_file is not None:
                desc_text = args.description_file.read_text(encoding="utf-8")
            _validate_file(args.input)
            response = upload_to_youtube(
                video_path=args.input,
                title=args.title,
                description=desc_text,
                tags=args.tags,
                category_id=args.category_id,
                privacy=args.privacy,
                thumbnail_path=args.thumbnail,
                client_secrets=args.client_secrets,
                token_path=args.token_path,
            )
            video_id = response["id"]
            print(f"Uploaded: https://www.youtube.com/watch?v={video_id}")
            return 0

        _validate_cli_options(args)
        require_binary("ffmpeg")
        require_binary("ffprobe")
        _validate_file(args.input)

        # ── analyze ───────────────────────────────────────────────────────
        if args.command == "analyze":
            stats = analyze_recording(
                input_path=args.input,
                plan_path=args.plan,
                scene_threshold=args.scene_threshold,
                clip_before=args.clip_before,
                clip_after=args.clip_after,
                min_gap_seconds=args.min_gap_seconds,
                max_clips=args.max_clips,
                target_duration_seconds=args.target_duration_seconds,
                target_duration_ratio=args.target_duration_ratio,
                intro_seconds=args.intro_seconds,
                outro_seconds=args.outro_seconds,
                vision_scoring=args.vision_scoring,
                vision_sample_fps=args.vision_sample_fps,
                vision_window_seconds=args.vision_window_seconds,
                vision_step_seconds=args.vision_step_seconds,
                whisper_model=args.whisper_model,
                whisper_bin=args.whisper_bin,
                whisper_language=args.whisper_language,
                whisper_threads=args.whisper_threads,
                whisper_audio_stream=args.whisper_audio_stream,
                whisper_vad=args.whisper_vad,
                whisper_vad_threshold=args.whisper_vad_threshold,
                whisper_vad_model=args.whisper_vad_model,
                ocr_cue_scoring=args.ocr_cue_scoring,
                tesseract_bin=args.tesseract_bin,
                ocr_sample_fps=args.ocr_sample_fps,
                ocr_cue_threshold=args.ocr_cue_threshold,
                ai_cue_threshold=args.ai_cue_threshold,
                end_on_result=args.end_on_result,
                result_detect_fps=args.result_detect_fps,
                result_detect_tail_seconds=args.result_detect_tail_seconds,
                one_shot_smart=args.one_shot_smart,
                cache=cache,
            )
            print(
                f"Wrote {args.plan} with {stats['segment_count']} segments "
                f"from {stats['event_count']} detected events "
                f"(ai-cues={stats['ai_cue_count']} "
                f"[transcript={stats['whisper_cue_count']}, ocr={stats['ocr_cue_count']}], "
                f"fallback={stats['used_fallback']}, "
                f"scene-threshold={stats['scene_threshold_used']:.3f})."
            )
            return 0

        # ── render ────────────────────────────────────────────────────────
        if args.command == "render":
            print("Rendering highlights...", file=sys.stderr, flush=True)
            render_highlights(
                input_path=args.input,
                plan_path=args.plan,
                output_path=args.output,
                crf=args.crf,
                preset=args.preset,
                video_encoder=args.video_encoder,
                allow_upscale=args.allow_upscale,
                auto_crop=args.auto_crop,
                crossfade_seconds=args.crossfade_seconds,
                audio_fade_seconds=args.audio_fade_seconds,
                two_pass_loudnorm=args.two_pass_loudnorm,
            )
            print(f"Rendered highlights: {args.output}")
            return 0

        # ── auto ──────────────────────────────────────────────────────────
        if args.command == "auto":
            def _run_analysis(
                plan_path: Path,
                overrides: dict[str, float | int] | None = None,
            ) -> dict[str, object]:
                opts = overrides or {}
                return analyze_recording(
                    input_path=args.input,
                    plan_path=plan_path,
                    scene_threshold=float(opts.get("scene_threshold", args.scene_threshold)),
                    clip_before=float(opts.get("clip_before", args.clip_before)),
                    clip_after=float(opts.get("clip_after", args.clip_after)),
                    min_gap_seconds=float(opts.get("min_gap_seconds", args.min_gap_seconds)),
                    max_clips=int(opts.get("max_clips", args.max_clips)),
                    target_duration_seconds=args.target_duration_seconds,
                    target_duration_ratio=float(
                        opts.get("target_duration_ratio", args.target_duration_ratio)
                    ),
                    intro_seconds=args.intro_seconds,
                    outro_seconds=args.outro_seconds,
                    vision_scoring=args.vision_scoring,
                    vision_sample_fps=args.vision_sample_fps,
                    vision_window_seconds=args.vision_window_seconds,
                    vision_step_seconds=args.vision_step_seconds,
                    whisper_model=args.whisper_model,
                    whisper_bin=args.whisper_bin,
                    whisper_language=args.whisper_language,
                    whisper_threads=args.whisper_threads,
                    whisper_audio_stream=args.whisper_audio_stream,
                    whisper_vad=args.whisper_vad,
                    whisper_vad_threshold=args.whisper_vad_threshold,
                    whisper_vad_model=args.whisper_vad_model,
                    ocr_cue_scoring=args.ocr_cue_scoring,
                    tesseract_bin=args.tesseract_bin,
                    ocr_sample_fps=args.ocr_sample_fps,
                    ocr_cue_threshold=float(opts.get("ocr_cue_threshold", args.ocr_cue_threshold)),
                    ai_cue_threshold=float(opts.get("ai_cue_threshold", args.ai_cue_threshold)),
                    end_on_result=args.end_on_result,
                    result_detect_fps=args.result_detect_fps,
                    result_detect_tail_seconds=args.result_detect_tail_seconds,
                    one_shot_smart=args.one_shot_smart,
                    cache=cache,
                )

            selected_stats: dict[str, object]
            selected_report: dict[str, object] | None = None

            if args.auto_optimize:
                variants = _build_auto_optimize_variants(
                    scene_threshold=args.scene_threshold,
                    clip_before=args.clip_before,
                    clip_after=args.clip_after,
                    min_gap_seconds=args.min_gap_seconds,
                    max_clips=args.max_clips,
                    target_duration_ratio=args.target_duration_ratio,
                    ai_cue_threshold=args.ai_cue_threshold,
                    ocr_cue_threshold=args.ocr_cue_threshold,
                    candidate_count=args.optimize_candidates,
                )
                if not variants:
                    raise EditorError("Auto-optimize did not generate any candidate variants.")
                print(
                    f"Auto-optimize: evaluating {len(variants)} candidates "
                    f"analytically (no rendering) by {args.optimize_metric} score.",
                    file=sys.stderr,
                    flush=True,
                )
                best_plan_data: dict[str, object] | None = None
                import tempfile
                with tempfile.TemporaryDirectory(prefix="lol-optimize-") as tmp:
                    tmp_path = Path(tmp)
                    for idx, variant in enumerate(variants, start=1):
                        candidate_plan = tmp_path / f"candidate-{idx}.json"
                        print(
                            f"  Candidate {idx}/{len(variants)}...",
                            file=sys.stderr,
                            flush=True,
                        )
                        candidate_stats = _run_analysis(candidate_plan, overrides=variant)
                        # Score analytically using signals already computed by analyze_recording
                        vision_windows = list(candidate_stats.get("_vision_windows") or [])
                        events = list(candidate_stats.get("_events") or [])
                        ai_cues = list(candidate_stats.get("_ai_cues") or [])
                        duration_seconds = float(candidate_stats.get("_duration_seconds") or 0.0)
                        segments = read_plan(candidate_plan)
                        candidate_report = score_plan_analytically(
                            segments,
                            vision_windows=vision_windows,
                            events=events,
                            duration_seconds=duration_seconds,
                            ai_cues=ai_cues or None,
                        )
                        metric_value = _select_optimize_metric_value(
                            report=candidate_report,
                            metric=args.optimize_metric,
                        )
                        print(
                            f"    youtube={candidate_report['youtube_score']}/100 | "
                            f"watchability={candidate_report['watchability_score']}/100 | "
                            f"quality={candidate_report['highlight_quality_score']}/100",
                            file=sys.stderr,
                            flush=True,
                        )
                        if best_plan_data is None or metric_value > float(
                            best_plan_data["metric_value"]
                        ):
                            best_plan_data = {
                                "metric_value": metric_value,
                                "plan_content": candidate_plan.read_bytes(),
                                "stats": candidate_stats,
                                "report": candidate_report,
                                "variant": variant,
                            }
                    if best_plan_data is None:
                        raise EditorError("Auto-optimize failed to evaluate candidates.")
                    args.plan.parent.mkdir(parents=True, exist_ok=True)
                    args.plan.write_bytes(bytes(best_plan_data["plan_content"]))
                    selected_stats = dict(best_plan_data["stats"])  # type: ignore[arg-type]
                    selected_report = dict(best_plan_data["report"])  # type: ignore[arg-type]
                    print(
                        f"Auto-optimize selected best candidate: "
                        f"{args.optimize_metric}={float(best_plan_data['metric_value']):.2f}.",
                        file=sys.stderr,
                        flush=True,
                    )
            else:
                selected_stats = _run_analysis(args.plan)

            print("Rendering highlights...", file=sys.stderr, flush=True)
            render_highlights(
                input_path=args.input,
                plan_path=args.plan,
                output_path=args.output,
                crf=args.crf,
                preset=args.preset,
                video_encoder=args.video_encoder,
                allow_upscale=args.allow_upscale,
                auto_crop=args.auto_crop,
                crossfade_seconds=args.crossfade_seconds,
                audio_fade_seconds=args.audio_fade_seconds,
                two_pass_loudnorm=args.two_pass_loudnorm,
            )
            print(
                f"Rendered {args.output} using {selected_stats['segment_count']} segments "
                f"(ai-cues={selected_stats['ai_cue_count']} "
                f"[transcript={selected_stats['whisper_cue_count']}, "
                f"ocr={selected_stats['ocr_cue_count']}], "
                f"plan: {args.plan})."
            )
            if selected_report is not None:
                print(
                    f"Auto-optimize estimate: "
                    f"youtube={selected_report['youtube_score']}/100 | "
                    f"watchability={selected_report['watchability_score']}/100 | "
                    f"quality={selected_report['highlight_quality_score']}/100",
                    file=sys.stderr,
                    flush=True,
                )
            return 0

        # ── full ──────────────────────────────────────────────────────────
        if args.command == "full":
            print("Rendering full match...", file=sys.stderr, flush=True)
            transcode_full_match(
                input_path=args.input,
                output_path=args.output,
                crf=args.crf,
                preset=args.preset,
                video_encoder=args.video_encoder,
                allow_upscale=args.allow_upscale,
                auto_crop=args.auto_crop,
                two_pass_loudnorm=args.two_pass_loudnorm,
            )
            print(f"Rendered full match: {args.output}")
            return 0

        # ── thumbnail ─────────────────────────────────────────────────────
        if args.command == "thumbnail":
            result = generate_thumbnail(
                input_path=args.input,
                output_path=args.output,
                timestamp_seconds=args.timestamp,
                scene_threshold=args.scene_threshold,
                vision_sample_fps=args.vision_sample_fps,
                vision_window_seconds=args.vision_window_seconds,
                vision_step_seconds=args.vision_step_seconds,
                width=args.width,
                height=args.height,
                quality=args.quality,
                auto_crop=args.auto_crop,
                enhance=args.enhance,
                champion_overlay_path=args.champion_overlay,
                champion_scale=args.champion_scale,
                champion_anchor=args.champion_anchor,
                headline_text=args.headline,
                headline_size=args.headline_size,
                headline_color=args.headline_color,
                headline_font=args.headline_font,
                headline_y_ratio=args.headline_y_ratio,
            )
            selection_mode = "auto" if bool(result["auto_selected"]) else "manual"
            extras: list[str] = []
            if bool(result.get("used_champion_overlay")):
                extras.append("champ-overlay")
            if bool(result.get("used_headline_text")):
                extras.append("headline")
            extras_suffix = f", extras={'+'.join(extras)}" if extras else ""
            print(
                f"Wrote thumbnail: {args.output} "
                f"(timestamp={result['timestamp_seconds']}s, mode={selection_mode}{extras_suffix})."
            )
            return 0

        # ── description ───────────────────────────────────────────────────
        if args.command == "description":
            package = generate_youtube_description(
                input_path=args.input,
                output_path=args.output,
                champion=args.champion,
                channel_name=args.channel_name,
                scene_threshold=args.scene_threshold,
                vision_sample_fps=args.vision_sample_fps,
                vision_window_seconds=args.vision_window_seconds,
                vision_step_seconds=args.vision_step_seconds,
                whisper_model=args.whisper_model,
                whisper_bin=args.whisper_bin,
                whisper_language=args.whisper_language,
                whisper_threads=args.whisper_threads,
                whisper_audio_stream=args.whisper_audio_stream,
                whisper_vad=args.whisper_vad,
                whisper_vad_threshold=args.whisper_vad_threshold,
                whisper_vad_model=args.whisper_vad_model,
                ocr_cue_scoring=args.ocr_cue_scoring,
                tesseract_bin=args.tesseract_bin,
                ocr_sample_fps=args.ocr_sample_fps,
                ocr_cue_threshold=args.ocr_cue_threshold,
                ai_cue_threshold=args.ai_cue_threshold,
                max_moments=args.max_moments,
                title_count=args.title_count,
            )
            titles = package.get("titles", [])
            top_title = titles[0] if isinstance(titles, list) and titles else ""
            print(f"Wrote description package: {args.output}")
            if top_title:
                print(f"Top title: {top_title}")
            if args.report is not None:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(package, indent=2), encoding="utf-8")
                print(f"Wrote description report: {args.report}")
            return 0

        # ── pipeline ─────────────────────────────────────────────────────
        if args.command == "pipeline":
            # Step 1: Analyze + Render (same as "auto")
            print("Step 1/4: Analyzing + rendering highlights...", file=sys.stderr, flush=True)
            analyze_stats = analyze_recording(
                input_path=args.input,
                plan_path=args.plan,
                scene_threshold=args.scene_threshold,
                clip_before=args.clip_before,
                clip_after=args.clip_after,
                min_gap_seconds=args.min_gap_seconds,
                max_clips=args.max_clips,
                target_duration_seconds=args.target_duration_seconds,
                target_duration_ratio=args.target_duration_ratio,
                intro_seconds=args.intro_seconds,
                outro_seconds=args.outro_seconds,
                vision_scoring=args.vision_scoring,
                vision_sample_fps=args.vision_sample_fps,
                vision_window_seconds=args.vision_window_seconds,
                vision_step_seconds=args.vision_step_seconds,
                whisper_model=args.whisper_model,
                whisper_bin=args.whisper_bin,
                whisper_language=args.whisper_language,
                whisper_threads=args.whisper_threads,
                whisper_audio_stream=args.whisper_audio_stream,
                whisper_vad=args.whisper_vad,
                whisper_vad_threshold=args.whisper_vad_threshold,
                whisper_vad_model=args.whisper_vad_model,
                ocr_cue_scoring=args.ocr_cue_scoring,
                tesseract_bin=args.tesseract_bin,
                ocr_sample_fps=args.ocr_sample_fps,
                ocr_cue_threshold=args.ocr_cue_threshold,
                ai_cue_threshold=args.ai_cue_threshold,
                end_on_result=args.end_on_result,
                result_detect_fps=args.result_detect_fps,
                result_detect_tail_seconds=args.result_detect_tail_seconds,
                one_shot_smart=args.one_shot_smart,
                cache=cache,
            )
            print(
                f"  Analysis: {analyze_stats['segment_count']} segments from "
                f"{analyze_stats['event_count']} events.",
                file=sys.stderr, flush=True,
            )
            render_highlights(
                input_path=args.input,
                plan_path=args.plan,
                output_path=args.output,
                crf=args.crf,
                preset=args.preset,
                video_encoder=args.video_encoder,
                allow_upscale=args.allow_upscale,
                auto_crop=args.auto_crop,
                crossfade_seconds=args.crossfade_seconds,
                audio_fade_seconds=args.audio_fade_seconds,
                two_pass_loudnorm=args.two_pass_loudnorm,
            )
            print(f"  Rendered: {args.output}", file=sys.stderr, flush=True)

            # Step 2: Thumbnail
            print("Step 2/4: Generating thumbnail...", file=sys.stderr, flush=True)
            # Auto-detect champion.png in project dir if not explicitly given
            champ_overlay = args.champion_overlay
            if champ_overlay is None:
                auto_champ = Path("champion.png")
                if auto_champ.exists():
                    champ_overlay = auto_champ
                    print(f"  Auto-detected champion overlay: {auto_champ}", file=sys.stderr, flush=True)
            thumb_result = generate_thumbnail(
                input_path=args.output,
                output_path=args.thumbnail_output,
                timestamp_seconds=None,
                scene_threshold=DEFAULT_THUMBNAIL_SCENE_THRESHOLD,
                vision_sample_fps=DEFAULT_THUMBNAIL_VISION_SAMPLE_FPS,
                vision_window_seconds=DEFAULT_THUMBNAIL_VISION_WINDOW_SECONDS,
                vision_step_seconds=DEFAULT_THUMBNAIL_VISION_STEP_SECONDS,
                width=DEFAULT_THUMBNAIL_WIDTH,
                height=DEFAULT_THUMBNAIL_HEIGHT,
                quality=DEFAULT_THUMBNAIL_QUALITY,
                auto_crop=True,
                enhance=True,
                champion_overlay_path=champ_overlay,
                champion_scale=DEFAULT_THUMBNAIL_CHAMPION_SCALE,
                champion_anchor=DEFAULT_THUMBNAIL_CHAMPION_ANCHOR,
                headline_text=args.headline,
                headline_size=DEFAULT_THUMBNAIL_HEADLINE_SIZE,
                headline_color=DEFAULT_THUMBNAIL_HEADLINE_COLOR,
                headline_font=args.headline_font,
                headline_y_ratio=DEFAULT_THUMBNAIL_HEADLINE_Y_RATIO,
            )
            print(f"  Thumbnail: {args.thumbnail_output}", file=sys.stderr, flush=True)

            # Step 3: Description
            print("Step 3/4: Generating description...", file=sys.stderr, flush=True)
            desc_package = generate_youtube_description(
                input_path=args.output,
                output_path=args.description_output,
                champion=args.champion,
                channel_name=args.channel_name,
                scene_threshold=args.scene_threshold,
                vision_sample_fps=args.vision_sample_fps,
                vision_window_seconds=args.vision_window_seconds,
                vision_step_seconds=args.vision_step_seconds,
                whisper_model=args.whisper_model,
                whisper_bin=args.whisper_bin,
                whisper_language=args.whisper_language,
                whisper_threads=args.whisper_threads,
                whisper_audio_stream=args.whisper_audio_stream,
                whisper_vad=args.whisper_vad,
                whisper_vad_threshold=args.whisper_vad_threshold,
                whisper_vad_model=args.whisper_vad_model,
                ocr_cue_scoring=args.ocr_cue_scoring,
                tesseract_bin=args.tesseract_bin,
                ocr_sample_fps=args.ocr_sample_fps,
                ocr_cue_threshold=args.ocr_cue_threshold,
                ai_cue_threshold=args.ai_cue_threshold,
            )
            titles = desc_package.get("titles", [])
            top_title = titles[0] if isinstance(titles, list) and titles else ""
            desc_text = desc_package.get("description_text", "")
            print(f"  Description: {args.description_output}", file=sys.stderr, flush=True)
            if top_title:
                print(f"  Title: {top_title}", file=sys.stderr, flush=True)

            # Step 4: Upload
            if args.upload:
                print("Step 4/4: Uploading to YouTube...", file=sys.stderr, flush=True)
                if not top_title:
                    top_title = args.output.stem.replace("-", " ").replace("_", " ").title()
                response = upload_to_youtube(
                    video_path=args.output,
                    title=top_title,
                    description=desc_text,
                    tags=args.tags or ["League of Legends", "LoL", "highlights"],
                    privacy=args.privacy,
                    thumbnail_path=args.thumbnail_output,
                    client_secrets=args.client_secrets,
                    token_path=args.token_path,
                )
                video_id = response["id"]
                print(f"Uploaded: https://www.youtube.com/watch?v={video_id}")
            else:
                print("Step 4/4: Upload skipped (--no-upload).", file=sys.stderr, flush=True)
                print(f"Pipeline complete. Video: {args.output}")
            return 0

        # ── watchability ──────────────────────────────────────────────────
        if args.command == "watchability":
            report = analyze_watchability(
                input_path=args.input,
                scene_threshold=args.scene_threshold,
                vision_sample_fps=args.vision_sample_fps,
                vision_window_seconds=args.vision_window_seconds,
                vision_step_seconds=args.vision_step_seconds,
            )
            print(
                f"Watchability score: {report['watchability_score']}/100 ({report['rating']})."
            )
            print(
                f"Highlight quality: {report['highlight_quality_score']}/100 "
                f"({report['quality_rating']})."
            )
            blend = report.get("score_blend", {})
            if isinstance(blend, dict):
                print(
                    f"YouTube score: {report['youtube_score']}/100 "
                    f"(watchability={blend.get('watchability_weight')}, "
                    f"quality={blend.get('quality_weight')})."
                )
            print(
                f"Events/min: {report['event_rate_per_minute']} | "
                f"Expected/min: {report['expected_event_rate_per_minute']} | "
                f"Profile: {report['activity_profile']} | "
                f"Scene threshold: {report['scene_threshold_used']} | "
                f"Low activity: {report['low_activity_ratio']} | "
                f"Gray-screen ratio: {report['gray_screen_ratio']}"
            )
            recommendations = report.get("recommendations", [])
            if isinstance(recommendations, list):
                for rec in recommendations:
                    if isinstance(rec, str):
                        print(f"- {rec}")
            if args.report is not None:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"Wrote watchability report: {args.report}")
            return 0

        parser.error(f"Unknown command: {args.command}")

    except EditorError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print("ffmpeg/ffprobe command failed:", file=sys.stderr)
        print(" ".join(error.cmd), file=sys.stderr)
        if error.stderr:
            print(error.stderr, file=sys.stderr)
        return error.returncode or 1
    except KeyboardInterrupt:
        print("Canceled by user.", file=sys.stderr)
        return 130
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
