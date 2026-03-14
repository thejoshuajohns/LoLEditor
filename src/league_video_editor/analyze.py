"""Analysis pipeline: scene detection, vision scoring, segment building.

This module owns all signal-extraction and segment-selection logic.  Every
expensive ffmpeg pass is wrapped with an optional :class:`~.cache.CacheStore`
so repeated ``analyze`` calls on the same file are near-instant.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
from collections import deque
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from .cache import (
    CacheStore,
    cropdetect_key,
    ocr_cues_key,
    result_time_key,
    scene_events_key,
    signalstats_key,
    whisper_key,
)
from .ffmpeg_utils import (
    detect_crop_filter,
    detect_scene_events,
    detect_scene_events_adaptive,
    extract_audio_for_whisper,
    extract_ocr_frames,
    extract_result_ocr_frames,
    extract_signalstats_samples,
    resolve_tesseract_binary,
    resolve_whisper_cpp_binary,
    run_command,
    run_tesseract_ocr,
)
from .models import (
    ADAPTIVE_AI_CUE_THRESHOLD_FACTORS,
    ADAPTIVE_AI_CUE_THRESHOLD_MIN,
    ADAPTIVE_SCENE_THRESHOLD_FACTORS,
    ADAPTIVE_SCENE_THRESHOLD_MIN,
    AI_WINDOW_BOOST_RADIUS_SECONDS,
    DEFAULT_AI_CUE_THRESHOLD,
    DEFAULT_INTRO_SECONDS,
    DEFAULT_OCR_CUE_SCORING,
    DEFAULT_OCR_CUE_THRESHOLD,
    DEFAULT_OCR_SAMPLE_FPS,
    DEFAULT_ONE_SHOT_SMART,
    DEFAULT_OUTRO_SECONDS,
    DEFAULT_RESULT_DETECT_FPS,
    DEFAULT_RESULT_DETECT_TAIL_SECONDS,
    DEFAULT_TARGET_DURATION_RATIO,
    DEFAULT_TARGET_DURATION_SECONDS,
    DEFAULT_TESSERACT_BIN,
    DEFAULT_VISION_SAMPLE_FPS,
    DEFAULT_VISION_SCORING,
    DEFAULT_VISION_STEP_SECONDS,
    DEFAULT_VISION_WINDOW_SECONDS,
    DEFAULT_WHISPER_CPP_BIN,
    DEFAULT_WHISPER_LANGUAGE,
    DEFAULT_WHISPER_THREADS,
    DEFAULT_WHISPER_VAD,
    DEFAULT_WHISPER_VAD_THRESHOLD,
    EVENT_CONTEXT_AI_AFTER_SECONDS,
    EVENT_CONTEXT_AI_BEFORE_SECONDS,
    EVENT_CONTEXT_COMBAT_AFTER_SECONDS,
    EVENT_CONTEXT_COMBAT_BEFORE_SECONDS,
    EVENT_CONTEXT_DEATH_AFTER_SECONDS,
    EVENT_CONTEXT_DEATH_BEFORE_SECONDS,
    TRANSCRIPT_HYPE_PATTERNS,
    WHISPER_TIMESTAMP_PATTERN,
    EditorError,
    Segment,
    TranscriptionCue,
    VisionWindow,
)


# ---------------------------------------------------------------------------
# Pure math helpers
# ---------------------------------------------------------------------------


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    clamped = min(1.0, max(0.0, percentile))
    if len(ordered) == 1:
        return ordered[0]
    position = clamped * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _normalize(value: float, *, low: float, high: float) -> float:
    span = high - low
    if span <= 1e-9:
        return 0.5
    return min(1.0, max(0.0, (value - low) / span))


def _coerce_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _average(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def _closeness_score(value: float, *, target: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(value - target) / tolerance)


# ---------------------------------------------------------------------------
# Transcript scoring
# ---------------------------------------------------------------------------


def _parse_timestamp_seconds(value: str) -> float | None:
    match = WHISPER_TIMESTAMP_PATTERN.match(value.strip())
    if not match:
        return None
    h, m, s, ms = match.groups()
    try:
        return max(
            0.0,
            int(h) * 3600 + int(m) * 60 + int(s) + int((ms or "0").ljust(3, "0")[:3]) / 1000.0,
        )
    except ValueError:
        return None


def _extract_whisper_segment_bounds(raw: dict[str, object]) -> tuple[float, float] | None:
    # offsets dict (ms)
    offsets = raw.get("offsets")
    if isinstance(offsets, dict):
        s = _coerce_float(offsets.get("from"))
        e = _coerce_float(offsets.get("to"))
        if s is not None and e is not None and e > s:
            return max(0.0, s / 1000.0), max(0.0, e / 1000.0)
    # timestamps dict
    timestamps = raw.get("timestamps")
    if isinstance(timestamps, dict):
        ts = timestamps.get("from")
        te = timestamps.get("to")
        if isinstance(ts, str) and isinstance(te, str):
            ss = _parse_timestamp_seconds(ts)
            es = _parse_timestamp_seconds(te)
            if ss is not None and es is not None and es > ss:
                return ss, es
    # direct start/end
    ds = _coerce_float(raw.get("start"))
    de = _coerce_float(raw.get("end"))
    if ds is not None and de is not None and de > ds:
        return max(0.0, ds), max(0.0, de)
    # t0/t1 in 10ms units
    t0 = _coerce_float(raw.get("t0"))
    t1 = _coerce_float(raw.get("t1"))
    if t0 is not None and t1 is not None and t1 > t0:
        return max(0.0, t0 / 100.0), max(0.0, t1 / 100.0)
    return None


def score_transcript_text(text: str) -> tuple[float, tuple[str, ...]]:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return 0.0, ()
    if re.fullmatch(r"(?:\[[^\]]+\]\s*)+", normalized):
        return 0.0, ()

    keyword_hits: list[str] = []
    raw_score = 0.0
    semantic_signal = False
    for phrase, weight, pattern in TRANSCRIPT_HYPE_PATTERNS:
        if pattern.search(normalized):
            keyword_hits.append(phrase)
            raw_score += weight
            semantic_signal = True

    regex_bonuses: tuple[tuple[str, float], ...] = (
        (r"\b(has|have)\s+slain\b", 0.24),
        (r"\b(you|we)\s+(killed|kill|got)\b", 0.16),
        (r"\b(enemy|ally)\s+(double|triple|quadra|penta)\b", 0.22),
        (r"\b(turret|tower)\s+(down|destroyed)\b", 0.18),
        (r"\b(steal|stole|stolen)\b", 0.14),
    )
    for pat, bonus in regex_bonuses:
        if re.search(pat, normalized):
            raw_score += bonus
            semantic_signal = True

    kill_mentions = len(re.findall(r"\bkill(?:ed|ing|s)?\b", normalized))
    if kill_mentions > 1:
        raw_score += min(0.35, (kill_mentions - 1) * 0.08)
    if kill_mentions > 0:
        semantic_signal = True
    slain_mentions = len(re.findall(r"\bslain\b", normalized))
    if slain_mentions > 0:
        raw_score += min(0.26, slain_mentions * 0.09)
        semantic_signal = True

    punct_boost = min(0.24, text.count("!") * 0.05)
    uppercase = sum(1 for c in text if c.isupper())
    alpha = sum(1 for c in text if c.isalpha())
    if alpha > 0 and uppercase / alpha >= 0.38 and alpha >= 8:
        raw_score += min(0.18, (uppercase / alpha) * 0.35)
    raw_score += punct_boost

    if not semantic_signal:
        return 0.0, ()
    score = max(0.0, min(1.0, math.tanh(raw_score * 0.72)))
    return score, tuple(sorted(set(keyword_hits)))


# ---------------------------------------------------------------------------
# Cue parsers (whisper JSON + OCR)
# ---------------------------------------------------------------------------


def _parse_whisper_json_cues(
    output_json_path: Path,
    *,
    cue_threshold: float,
) -> list[TranscriptionCue]:
    try:
        data = json.loads(output_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise EditorError(f"Could not parse whisper transcript JSON: {output_json_path}") from err

    raw_segments: list[object] = []
    if isinstance(data, dict):
        for key in ("transcription", "segments"):
            candidate = data.get(key)
            if isinstance(candidate, list):
                raw_segments = candidate
                break
        if not raw_segments:
            result = data.get("result")
            if isinstance(result, dict):
                for key in ("segments", "transcription"):
                    candidate = result.get(key)
                    if isinstance(candidate, list):
                        raw_segments = candidate
                        break

    cues: list[TranscriptionCue] = []
    for raw in raw_segments:
        if not isinstance(raw, dict):
            continue
        bounds = _extract_whisper_segment_bounds(raw)
        if bounds is None:
            continue
        start, end = bounds
        text_value = raw.get("text", "")
        text = text_value.strip() if isinstance(text_value, str) else str(text_value).strip()
        if not text:
            continue
        score, kw = score_transcript_text(text)
        if score < cue_threshold:
            continue
        cues.append(TranscriptionCue(start=start, end=end, score=score, text=text, keywords=kw))

    cues.sort(key=lambda c: (c.start, -c.score))
    deduped: list[TranscriptionCue] = []
    for cue in cues:
        if deduped and cue.start - deduped[-1].start < 0.75:
            if cue.score > deduped[-1].score:
                deduped[-1] = cue
            continue
        deduped.append(cue)
    return deduped


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
    _resolve_binary_fn=None,
    _extract_audio_fn=None,
    _run_cmd_fn=None,
) -> list[TranscriptionCue]:
    _resolve = _resolve_binary_fn if _resolve_binary_fn is not None else resolve_whisper_cpp_binary
    _extract = _extract_audio_fn if _extract_audio_fn is not None else extract_audio_for_whisper
    _run = _run_cmd_fn if _run_cmd_fn is not None else run_command

    if not whisper_model.exists() or not whisper_model.is_file():
        raise EditorError(f"Whisper model file not found: {whisper_model}")
    resolved = _resolve(whisper_binary)
    if resolved is None:
        raise EditorError(
            "Could not find whisper.cpp binary. Install whisper.cpp and ensure "
            "`whisper-cli` is on PATH, or pass --whisper-bin with a full path."
        )
    with tempfile.TemporaryDirectory(prefix="lol-whisper-") as tmp:
        tmp_path = Path(tmp)
        audio_path = tmp_path / "audio.wav"
        try:
            _extract(
                input_path=input_path,
                output_audio_path=audio_path,
                audio_stream_index=whisper_audio_stream,
            )
        except subprocess.CalledProcessError as err:
            detail = f"\n{(err.stderr or '').strip()}" if err.stderr else ""
            raise EditorError(f"Could not extract audio for whisper.{detail}") from err

        output_prefix = tmp_path / "transcript"
        cmd = [
            resolved,
            "-m",
            str(whisper_model),
            "-f",
            str(audio_path),
            "-oj",
            "-of",
            str(output_prefix),
            "-t",
            str(max(1, whisper_threads)),
        ]
        if whisper_language:
            cmd.extend(["-l", whisper_language])
        if whisper_vad and whisper_vad_model is not None:
            if not whisper_vad_model.exists():
                raise EditorError(f"Whisper VAD model not found: {whisper_vad_model}")
            cmd.extend(
                [
                    "--vad",
                    "--vad-model",
                    str(whisper_vad_model),
                    "--vad-threshold",
                    f"{whisper_vad_threshold:.3f}",
                ]
            )
        result = _run(cmd, capture_output=True)
        output_json_path = Path(f"{output_prefix}.json")
        if not output_json_path.exists():
            candidates = sorted(tmp_path.glob("*.json"))
            if candidates:
                output_json_path = candidates[0]
            else:
                whisper_out = ((result.stderr or "") + "\n" + (result.stdout or "")).strip()
                tail = whisper_out[-1200:] if whisper_out else ""
                detail = f"\nwhisper-cli output:\n{tail}" if tail else ""
                raise EditorError(
                    "whisper.cpp did not produce a JSON transcript. "
                    f"Ensure your whisper-cli supports --output-json.{detail}"
                )
        return _parse_whisper_json_cues(output_json_path, cue_threshold=cue_threshold)


def _collect_ocr_cues(
    *,
    input_path: Path,
    tesseract_binary: str,
    sample_fps: float,
    cue_threshold: float,
    _resolve_tesseract_fn=None,
    _extract_frames_fn=None,
    _run_ocr_fn=None,
) -> list[TranscriptionCue]:
    _resolve = _resolve_tesseract_fn if _resolve_tesseract_fn is not None else resolve_tesseract_binary
    _extract = _extract_frames_fn if _extract_frames_fn is not None else extract_ocr_frames
    _run_ocr = _run_ocr_fn if _run_ocr_fn is not None else run_tesseract_ocr

    resolved = _resolve(tesseract_binary)
    if resolved is None:
        raise EditorError("Could not find tesseract. Install or pass --tesseract-bin.")
    if sample_fps <= 0:
        return []
    with tempfile.TemporaryDirectory(prefix="lol-ocr-") as tmp:
        frames_dir = Path(tmp) / "frames"
        frame_paths = _extract(
            input_path=input_path,
            output_dir=frames_dir,
            sample_fps=sample_fps,
        )
        if not frame_paths:
            return []
        cues: list[TranscriptionCue] = []
        for index, frame_path in enumerate(frame_paths):
            text = _run_ocr(image_path=frame_path, tesseract_binary=resolved)
            if not text:
                continue
            score, kw = score_transcript_text(text)
            if score < cue_threshold:
                continue
            center = index / sample_fps
            start = max(0.0, center - 0.8)
            end = center + 2.2
            cues.append(TranscriptionCue(start=start, end=end, score=score, text=text, keywords=kw))
    cues.sort(key=lambda c: (c.start, -c.score))
    deduped: list[TranscriptionCue] = []
    for cue in cues:
        if deduped and cue.start - deduped[-1].start < 8.0:
            if cue.score > deduped[-1].score:
                deduped[-1] = cue
            continue
        deduped.append(cue)
    return deduped


def _detect_terminal_result_time(
    *,
    input_path: Path,
    duration_seconds: float,
    tesseract_binary: str,
    sample_fps: float = DEFAULT_RESULT_DETECT_FPS,
    tail_seconds: float = DEFAULT_RESULT_DETECT_TAIL_SECONDS,
) -> float | None:
    if duration_seconds <= 0 or sample_fps <= 0:
        return None
    resolved = resolve_tesseract_binary(tesseract_binary)
    if resolved is None:
        return None
    start_time = max(0.0, duration_seconds - max(60.0, tail_seconds))
    with tempfile.TemporaryDirectory(prefix="lol-result-ocr-") as tmp:
        frames_dir = Path(tmp) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_paths = extract_result_ocr_frames(
            input_path=input_path,
            output_dir=frames_dir,
            sample_fps=sample_fps,
            start_time=start_time,
        )
        if not frame_paths:
            return None
        result_hits: list[float] = []
        nexus_hits: list[float] = []
        result_pattern = re.compile(r"\b(victory|defeat)\b")
        nexus_pattern = re.compile(r"\bnexus\b")
        terminal_nexus_pattern = re.compile(
            r"\b(nexus\b.*\b(destroyed|falls|fall|down|explodes|exploded)\b"
            r"|destroyed\b.*\bnexus\b)"
        )
        for index, frame_path in enumerate(frame_paths):
            text = run_tesseract_ocr(image_path=frame_path, tesseract_binary=resolved)
            if not text:
                continue
            norm = re.sub(r"\s+", " ", text.lower())
            frame_time = start_time + (index / sample_fps)
            if result_pattern.search(norm):
                result_hits.append(frame_time)
            if nexus_pattern.search(norm) and terminal_nexus_pattern.search(norm):
                nexus_hits.append(frame_time)
    if result_hits:
        return min(duration_seconds, result_hits[-1] + 2.5)
    if nexus_hits:
        return min(duration_seconds, nexus_hits[-1] + 4.0)
    return None


# ---------------------------------------------------------------------------
# Vision scoring
# ---------------------------------------------------------------------------


def _compute_vision_window_scores(
    *,
    frame_samples: list[tuple[float, float, float]],
    events: list[float],
    duration_seconds: float,
    window_seconds: float,
    step_seconds: float,
) -> list[VisionWindow]:
    if not frame_samples or duration_seconds <= 0 or window_seconds <= 0 or step_seconds <= 0:
        return []

    sorted_events = sorted(v for v in events if v >= 0)
    windows_raw: list[tuple[float, float, float, float, float]] = []
    max_start = max(0.0, duration_seconds - window_seconds)
    start = 0.0
    while start <= max_start + 1e-6:
        end = min(duration_seconds, start + window_seconds)
        in_window = [
            s
            for s in frame_samples
            if start <= s[0] < end and math.isfinite(s[1]) and math.isfinite(s[2])
        ]
        if in_window:
            motion = sum(s[1] for s in in_window) / len(in_window)
            saturation = sum(s[2] for s in in_window) / len(in_window)
            scene_hits = sum(1 for e in sorted_events if start <= e < end)
            scene_density = scene_hits / max(1.0, window_seconds)
            windows_raw.append((start, end, motion, saturation, scene_density))
        start += step_seconds

    if not windows_raw:
        return []

    motions = [w[2] for w in windows_raw]
    saturations = [w[3] for w in windows_raw]
    densities = [w[4] for w in windows_raw]
    motion_low = _percentile(motions, 0.10)
    motion_high = _percentile(motions, 0.90)
    density_high = max(densities) if any(d > 0 for d in densities) else 1.0
    sat_low = _percentile(saturations, 0.15)
    sat_high = _percentile(saturations, 0.90)
    low_motion_threshold = _percentile(motions, 0.25)
    low_density_threshold = _percentile(densities, 0.25)
    low_saturation_threshold = _percentile(saturations, 0.20)

    scored: list[VisionWindow] = []
    for start, end, motion, saturation, scene_density in windows_raw:
        motion_norm = _normalize(motion, low=motion_low, high=motion_high)
        scene_norm = _normalize(scene_density, low=0.0, high=density_high)
        sat_norm = _normalize(saturation, low=sat_low, high=sat_high)
        late_game_weight = 0.9 + 0.2 * (end / duration_seconds)
        score = (0.50 * motion_norm + 0.35 * scene_norm + 0.15 * sat_norm) * late_game_weight
        if motion <= low_motion_threshold and scene_density <= low_density_threshold:
            score -= 0.10
        if saturation <= low_saturation_threshold:
            score -= 0.12
        score = min(1.0, max(0.0, score))
        scored.append(
            VisionWindow(
                start=start,
                end=end,
                score=score,
                motion=motion,
                saturation=saturation,
                scene_density=scene_density,
            )
        )
    return scored


def score_vision_activity(
    input_path: Path,
    *,
    events: list[float],
    duration_seconds: float,
    sample_fps: float,
    window_seconds: float,
    step_seconds: float,
    show_progress: bool,
    progress_label: str | None = "Scoring gameplay",
    progress_callback: Callable[[float], None] | None = None,
    crop_filter: str | None = None,
) -> list[VisionWindow]:
    samples = extract_signalstats_samples(
        input_path,
        duration_seconds=duration_seconds,
        sample_fps=sample_fps,
        crop_filter=crop_filter,
        show_progress=show_progress,
        progress_label=progress_label,
        progress_callback=progress_callback,
    )
    return _compute_vision_window_scores(
        frame_samples=samples,
        events=events,
        duration_seconds=duration_seconds,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )


# ---------------------------------------------------------------------------
# AI-cue / vision-window boosting
# ---------------------------------------------------------------------------


def _boost_vision_windows_with_ai_cues(
    vision_windows: list[VisionWindow],
    ai_cues: list[TranscriptionCue],
    *,
    radius_seconds: float = AI_WINDOW_BOOST_RADIUS_SECONDS,
) -> list[VisionWindow]:
    if not vision_windows or not ai_cues:
        return vision_windows
    clamped_radius = max(1.0, radius_seconds)
    boosted: list[VisionWindow] = []
    for window in vision_windows:
        window_duration = max(0.001, window.end - window.start)
        window_center = (window.start + window.end) / 2
        best = 0.0
        for cue in ai_cues:
            cc = cue.center
            if cc < window.start - clamped_radius or cc > window.end + clamped_radius:
                continue
            overlap = max(0.0, min(window.end, cue.end) - max(window.start, cue.start))
            overlap_ratio = overlap / window_duration
            proximity = max(0.0, 1.0 - abs(window_center - cc) / clamped_radius)
            contribution = cue.score * (0.72 * proximity + 0.28 * overlap_ratio)
            if contribution > best:
                best = contribution
        boosted_score = min(1.0, window.score + 0.38 * best)
        boosted.append(
            VisionWindow(
                start=window.start,
                end=window.end,
                score=boosted_score,
                motion=window.motion,
                saturation=window.saturation,
                scene_density=window.scene_density,
            )
        )
    return boosted


def _extract_ai_priority_cues(ai_cues: list[TranscriptionCue]) -> list[float]:
    return sorted(
        {
            cue.center
            for cue in ai_cues
            if math.isfinite(cue.center) and cue.center >= 0.0
        }
    )


# ---------------------------------------------------------------------------
# Death / combat cue detection
# ---------------------------------------------------------------------------


def _detect_death_cues(
    vision_windows: list[VisionWindow],
    *,
    min_spacing_seconds: float = 35.0,
) -> list[float]:
    if len(vision_windows) < 2:
        return []
    ordered = sorted(vision_windows, key=lambda w: w.start)
    saturations = [w.saturation for w in ordered]
    motions = [w.motion for w in ordered]
    densities = [w.scene_density for w in ordered]
    sat_p30 = _percentile(saturations, 0.30)
    gray_threshold = max(7.0, min(18.0, sat_p30 + 2.0))
    motion_soft_cap = _percentile(motions, 0.65)
    density_soft_cap = _percentile(densities, 0.60)
    score_soft_cap = _percentile([w.score for w in ordered], 0.55)

    def looks_dead(w: VisionWindow) -> bool:
        return (
            w.saturation <= gray_threshold
            and w.score <= score_soft_cap
            and (w.motion <= motion_soft_cap or w.scene_density <= density_soft_cap)
        )

    death_cues: list[float] = []
    for i in range(1, len(ordered)):
        prev = ordered[i - 1]
        cur = ordered[i]
        if looks_dead(prev) or not looks_dead(cur):
            continue
        has_persistence = i + 1 < len(ordered) and looks_dead(ordered[i + 1])
        if not has_persistence:
            continue
        lead_in = max(10.0, (cur.end - cur.start) * 0.75)
        cue_time = max(0.0, cur.start - lead_in)
        if death_cues and cue_time - death_cues[-1] < min_spacing_seconds:
            continue
        death_cues.append(cue_time)
    return death_cues


def _detect_combat_cues(
    vision_windows: list[VisionWindow],
    *,
    min_spacing_seconds: float = 28.0,
) -> list[float]:
    if len(vision_windows) < 3:
        return []
    ordered = sorted(vision_windows, key=lambda w: w.start)
    scores = [w.score for w in ordered]
    motions = [w.motion for w in ordered]
    densities = [w.scene_density for w in ordered]
    saturations = [w.saturation for w in ordered]
    score_threshold = _percentile(scores, 0.68)
    motion_threshold = _percentile(motions, 0.68)
    density_threshold = _percentile(densities, 0.62)
    min_color_threshold = max(10.0, _percentile(saturations, 0.35))

    combat_cues: list[float] = []
    for index, current in enumerate(ordered):
        is_active = (
            current.score >= score_threshold
            and (current.motion >= motion_threshold or current.scene_density >= density_threshold)
            and current.saturation >= min_color_threshold
        )
        if not is_active:
            continue
        previous = ordered[index - 1] if index > 0 else None
        next_w = ordered[index + 1] if index + 1 < len(ordered) else None
        has_rise = previous is None or previous.score < current.score * 0.92 or previous.motion < motion_threshold * 0.9
        has_follow = next_w is None or next_w.score >= score_threshold * 0.72 or next_w.motion >= motion_threshold * 0.78
        if not has_rise or not has_follow:
            continue
        lead_in = max(12.0, (current.end - current.start) * 0.9)
        cue_time = max(0.0, current.start - lead_in)
        if combat_cues and cue_time - combat_cues[-1] < min_spacing_seconds:
            continue
        combat_cues.append(cue_time)
    return combat_cues


# ---------------------------------------------------------------------------
# Gameplay-start detection
# ---------------------------------------------------------------------------


def _detect_gameplay_start(
    vision_windows: list[VisionWindow],
    *,
    duration_seconds: float,
) -> float:
    if not vision_windows or duration_seconds <= 0:
        return 0.0
    ordered = sorted(vision_windows, key=lambda w: w.start)
    search_limit = min(duration_seconds * 0.6, 900.0)
    candidates = [w for w in ordered if w.start <= search_limit] or ordered
    scores = [w.score for w in candidates]
    motions = [w.motion for w in candidates]
    saturations = [w.saturation for w in candidates]
    densities = [w.scene_density for w in candidates]
    early_count = min(len(candidates), 12)
    baseline_motion = _percentile(motions[:early_count], 0.50) if motions else _percentile(motions, 0.25)
    baseline_score = _percentile(scores[:early_count], 0.50) if scores else _percentile(scores, 0.25)
    score_jump = baseline_score + 0.04 if baseline_score <= 0.20 else 0.0
    motion_jump = baseline_motion * 1.8 + 0.15 if baseline_motion <= 2.0 else 0.0
    score_threshold = max(0.10, _percentile(scores, 0.30), score_jump)
    motion_threshold = max(0.45, _percentile(motions, 0.25), motion_jump)
    saturation_threshold = max(8.0, _percentile(saturations, 0.20))
    density_threshold = max(0.002, _percentile(densities, 0.20))
    for index, window in enumerate(candidates):
        is_active = (
            window.score >= score_threshold
            and window.motion >= motion_threshold
            and window.saturation >= saturation_threshold
            and window.scene_density >= density_threshold
        )
        if not is_active:
            continue
        if index + 1 < len(candidates):
            nw = candidates[index + 1]
            if not (nw.score >= score_threshold * 0.7 or nw.motion >= motion_threshold * 0.8):
                continue
        return max(0.0, window.start - 3.0)
    return 0.0


# ---------------------------------------------------------------------------
# Segment helpers
# ---------------------------------------------------------------------------


def sample_evenly(values: list[float], max_items: int) -> list[float]:
    if max_items <= 0:
        return []
    if len(values) <= max_items:
        return values[:]
    if max_items == 1:
        return [values[len(values) // 2]]
    sampled = [values[round(i * (len(values) - 1) / (max_items - 1))] for i in range(max_items)]
    deduped = list(dict.fromkeys(sampled))
    if len(deduped) == max_items:
        return deduped
    for v in values:
        if len(deduped) == max_items:
            break
        if v not in deduped:
            deduped.append(v)
    return sorted(deduped)


def merge_segments(segments: list[Segment], merge_gap: float = 0.25) -> list[Segment]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: s.start)
    merged = [ordered[0]]
    for seg in ordered[1:]:
        last = merged[-1]
        if seg.start <= last.end + merge_gap:
            merged[-1] = Segment(last.start, max(last.end, seg.end))
        else:
            merged.append(seg)
    return merged


def _rank_vision_candidates(
    vision_windows: list[VisionWindow],
    *,
    min_gap_seconds: float,
    clip_before: float,
    clip_after: float,
) -> list[float]:
    if not vision_windows:
        return []
    ranked = sorted(vision_windows, key=lambda w: (-w.score, w.start))
    selected: list[float] = []
    spacing = max(min_gap_seconds, (clip_before + clip_after) * 0.25)
    for window in ranked:
        center = (window.start + window.end) / 2
        if any(abs(center - v) < spacing for v in selected):
            continue
        selected.append(center)
    return selected


def _rank_event_candidates(
    events: list[float],
    *,
    min_gap_seconds: float,
    duration_seconds: float,
    clip_before: float,
    clip_after: float,
) -> list[float]:
    filtered: list[float] = []
    for event in sorted(events):
        if event < 0:
            continue
        if not filtered or event - filtered[-1] >= min_gap_seconds:
            filtered.append(event)
    if not filtered:
        return []
    density_window = max(min_gap_seconds, clip_before + clip_after, 15.0)
    scored: list[tuple[float, float]] = []
    for event in filtered:
        local_density = sum(1 for other in filtered if abs(other - event) <= density_window)
        late_weight = 1.0 + (event / duration_seconds if duration_seconds > 0 else 0.0)
        scored.append((local_density * late_weight, event))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [e for _, e in scored]


def _generate_fallback_centers(
    *,
    start: float,
    end: float,
    count: int,
    min_gap_seconds: float,
    existing: list[float],
) -> list[float]:
    if count <= 0 or end <= start:
        return []
    selected: list[float] = []
    grid_count = max(12, count * 4)
    step = (end - start) / (grid_count + 1)
    for i in range(1, grid_count + 1):
        center = start + step * i
        if any(abs(center - v) < min_gap_seconds for v in existing + selected):
            continue
        selected.append(center)
        if len(selected) >= count:
            return selected
    remaining = count - len(selected)
    if remaining <= 0:
        return selected
    if remaining == 1:
        selected.append((start + end) / 2)
        return selected
    spacing = (end - start) / (remaining + 1)
    for i in range(remaining):
        selected.append(start + spacing * (i + 1))
    return selected


def _build_segment_from_center(
    center: float,
    *,
    clip_before: float,
    clip_after: float,
    clamp_start: float,
    clamp_end: float,
) -> Segment | None:
    start = max(clamp_start, center - clip_before)
    end = min(clamp_end, center + clip_after)
    if end <= start:
        return None
    return Segment(start=start, end=end)


def _derive_outro_segment(
    *,
    duration_seconds: float,
    events: list[float],
    outro_seconds: float,
    force_end_at_duration: bool = False,
) -> Segment | None:
    if duration_seconds <= 0 or outro_seconds <= 0:
        return None
    likely_finish = duration_seconds
    if not force_end_at_duration and events:
        last_event = max(events)
        if duration_seconds - last_event <= 180.0:
            likely_finish = min(duration_seconds, max(0.0, last_event))
    outro_end = max(0.0, likely_finish)
    outro_start = max(0.0, outro_end - outro_seconds)
    if outro_end <= outro_start:
        return None
    return Segment(start=outro_start, end=outro_end)


# ---------------------------------------------------------------------------
# Coverage constraint: ensure early / mid / late band representation
# ---------------------------------------------------------------------------


def _ensure_coverage_bands(
    segments: list[Segment],
    *,
    interior_start: float,
    interior_end: float,
    vision_windows: list[VisionWindow],
    clip_before: float,
    clip_after: float,
) -> tuple[list[Segment], list[str]]:
    """Add at most one segment per uncovered game-phase band.

    Returns the extended segment list and a list of band labels that were
    added (for plan metadata / explanation).
    """
    span = interior_end - interior_start
    if span <= 0:
        return segments, []

    bands = [
        ("early", interior_start, interior_start + span / 3),
        ("mid", interior_start + span / 3, interior_start + 2 * span / 3),
        ("late", interior_start + 2 * span / 3, interior_end),
    ]
    added_bands: list[str] = []
    result = list(segments)

    for band_name, band_start, band_end in bands:
        # Check if any existing segment overlaps this band
        has_coverage = any(s.start < band_end and s.end > band_start for s in result)
        if has_coverage:
            continue

        # Pick the highest-scoring vision window in this band
        band_windows = [
            w for w in vision_windows if w.start < band_end and w.end > band_start
        ]
        if band_windows:
            best_window = max(band_windows, key=lambda w: w.score)
            center = (best_window.start + best_window.end) / 2
        else:
            # Fallback: geometric center of band
            center = (band_start + band_end) / 2

        seg = _build_segment_from_center(
            center,
            clip_before=clip_before,
            clip_after=clip_after,
            clamp_start=interior_start,
            clamp_end=interior_end,
        )
        if seg is not None:
            result.append(seg)
            added_bands.append(band_name)

    return result, added_bands


# ---------------------------------------------------------------------------
# One-shot smart tuning
# ---------------------------------------------------------------------------


def _derive_one_shot_segment_tuning(
    *,
    events: list[float],
    duration_seconds: float,
    vision_windows: list[VisionWindow],
    ai_cues: list[TranscriptionCue],
    clip_before: float,
    clip_after: float,
    min_gap_seconds: float,
    max_clips: int,
    forced_cue_share: float,
) -> dict[str, float | int]:
    duration_minutes = max(0.1, duration_seconds / 60.0)
    event_rate = len(events) / duration_minutes
    ai_rate = len(ai_cues) / duration_minutes
    scores = [w.score for w in vision_windows]
    if scores:
        high_threshold = _percentile(scores, 0.74)
        low_threshold = _percentile(scores, 0.32)
        high_activity_ratio = sum(1 for s in scores if s >= high_threshold) / len(scores)
        low_activity_ratio = sum(1 for s in scores if s <= low_threshold) / len(scores)
    else:
        high_activity_ratio = 0.0
        low_activity_ratio = 1.0

    tuned_clip_before = clip_before
    tuned_clip_after = clip_after
    tuned_gap = min_gap_seconds
    tuned_max_clips = max_clips
    tuned_forced_share = forced_cue_share

    if ai_rate >= 0.45 or high_activity_ratio >= 0.28:
        tuned_clip_before += 1.5
        tuned_clip_after += 2.5
        tuned_forced_share += 0.10
    if ai_rate >= 0.70 or event_rate >= 2.4:
        tuned_gap -= 2.0
        tuned_max_clips += 4
        tuned_forced_share += 0.10
    elif low_activity_ratio >= 0.58:
        tuned_gap += 1.0
        tuned_max_clips -= 1
        tuned_forced_share -= 0.05
    if ai_rate < 0.15 and high_activity_ratio < 0.18:
        tuned_forced_share -= 0.08

    return {
        "clip_before": max(4.0, min(24.0, tuned_clip_before)),
        "clip_after": max(6.0, min(32.0, tuned_clip_after)),
        "min_gap_seconds": max(6.0, min(30.0, tuned_gap)),
        "max_clips": int(max(12, min(44, tuned_max_clips))),
        "forced_cue_share": max(0.35, min(0.85, tuned_forced_share)),
    }


# ---------------------------------------------------------------------------
# Segment builder (main entry point)
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
    _death_cue_fn=None,
    _combat_cue_fn=None,
) -> tuple[list[Segment], bool, list[str]]:
    """Build highlight segments.

    Returns
    -------
    (segments, used_fallback, coverage_bands_added)
        ``coverage_bands_added`` is the list of game-phase band labels
        (``"early"``, ``"mid"``, ``"late"``) that were filled in by the
        coverage-constraint pass.
    """
    if duration_seconds <= 0 or max_clips <= 0:
        return [], False, []

    gameplay_start = (
        _detect_gameplay_start(vision_windows or [], duration_seconds=duration_seconds)
        if vision_windows
        else 0.0
    )
    intro_start = min(duration_seconds, max(0.0, gameplay_start))
    intro_end = min(duration_seconds, intro_start + max(0.0, intro_seconds))
    intro_segment = Segment(start=intro_start, end=intro_end) if intro_end > intro_start else None
    outro_segment = _derive_outro_segment(
        duration_seconds=duration_seconds,
        events=events,
        outro_seconds=max(0.0, outro_seconds),
        force_end_at_duration=force_outro_to_duration_end,
    )
    anchored_segments = [s for s in (intro_segment, outro_segment) if s is not None]
    if (
        intro_segment is not None
        and outro_segment is not None
        and intro_segment.end >= outro_segment.start
    ):
        return [Segment(intro_segment.start, duration_seconds)], False, []

    interior_start = intro_segment.end if intro_segment is not None else 0.0
    interior_end = outro_segment.start if outro_segment is not None else duration_seconds
    clip_window_seconds = clip_before + clip_after
    target_total = duration_seconds if target_duration_seconds <= 0 else min(duration_seconds, target_duration_seconds)
    anchored_duration = sum(s.duration for s in anchored_segments)
    target_middle = max(0.0, target_total - anchored_duration)

    middle_clip_target = 0
    if clip_window_seconds > 0 and interior_end > interior_start and target_middle > 0:
        middle_clip_target = min(max_clips, math.ceil(target_middle / clip_window_seconds))

    middle_segments: list[Segment] = []
    used_fallback = False
    spacing_seconds = min_gap_seconds
    death_cue_detector = _death_cue_fn if _death_cue_fn is not None else _detect_death_cues
    combat_cue_detector = _combat_cue_fn if _combat_cue_fn is not None else _detect_combat_cues
    death_cues = death_cue_detector(vision_windows or [])
    combat_cues = combat_cue_detector(vision_windows or [])
    ai_cue_details: list[tuple[float, float]] = []
    if ai_priority_details:
        for cue in ai_priority_details:
            center = cue.center
            if math.isfinite(center) and center >= 0.0:
                ai_cue_details.append((center, max(0.0, min(1.0, cue.score))))
    elif ai_priority_cues:
        ai_cue_details.extend(
            (t, 0.40) for t in ai_priority_cues if math.isfinite(t) and t >= 0.0
        )
    ai_cue_details.sort(key=lambda c: c[0])
    forced_cues = sorted(
        [("death", t, 0.92) for t in death_cues]
        + [("combat", t, 0.86) for t in combat_cues]
        + [("ai", t, 0.55 + 0.45 * sc) for t, sc in ai_cue_details],
        key=lambda c: c[1],
    )

    if middle_clip_target > 0:
        effective_clip_before = clip_before
        effective_clip_after = clip_after
        if clip_window_seconds > 0 and target_middle > 0:
            target_per_clip = target_middle / middle_clip_target
            if target_per_clip > clip_window_seconds:
                expansion = min(1.35, target_per_clip / clip_window_seconds)
                effective_clip_before *= expansion
                effective_clip_after *= expansion

        ranked_events = (
            _rank_vision_candidates(
                vision_windows,
                min_gap_seconds=min_gap_seconds,
                clip_before=effective_clip_before,
                clip_after=effective_clip_after,
            )
            if vision_windows
            else _rank_event_candidates(
                events,
                min_gap_seconds=min_gap_seconds,
                duration_seconds=duration_seconds,
                clip_before=effective_clip_before,
                clip_after=effective_clip_after,
            )
        )

        forced_centers: list[float] = []
        forced_segments: list[Segment] = []
        candidate_forced_cues = forced_cues
        norm_share = min(0.95, max(0.20, forced_cue_share))
        max_forced = max(1, math.ceil(middle_clip_target * norm_share))
        if len(candidate_forced_cues) > max_forced:
            idx_list = sample_evenly(list(range(len(candidate_forced_cues))), max_forced)
            selected_forced = [candidate_forced_cues[i] for i in idx_list]
            priority_ranked = sorted(candidate_forced_cues, key=lambda c: (-c[2], c[1]))
            for cue in priority_ranked:
                if cue in selected_forced:
                    continue
                weakest = min(range(len(selected_forced)), key=lambda i: selected_forced[i][2])
                if cue[2] > selected_forced[weakest][2] + 0.05:
                    selected_forced[weakest] = cue
            candidate_forced_cues = sorted(selected_forced, key=lambda c: c[1])
        if len(candidate_forced_cues) > middle_clip_target:
            idx_list = sample_evenly(list(range(len(candidate_forced_cues))), middle_clip_target)
            candidate_forced_cues = [candidate_forced_cues[i] for i in idx_list]

        forced_spacing = max(6.0, spacing_seconds * 0.72)
        for cue_type, cue_time, cue_priority in candidate_forced_cues:
            if cue_time <= interior_start or cue_time >= interior_end:
                continue
            if any(abs(cue_time - c) < forced_spacing for c in forced_centers):
                continue
            if cue_type == "death":
                cb = max(effective_clip_before, EVENT_CONTEXT_DEATH_BEFORE_SECONDS)
                ca = max(effective_clip_after, EVENT_CONTEXT_DEATH_AFTER_SECONDS)
            elif cue_type == "combat":
                cb = max(effective_clip_before, EVENT_CONTEXT_COMBAT_BEFORE_SECONDS)
                ca = max(effective_clip_after, EVENT_CONTEXT_COMBAT_AFTER_SECONDS)
            else:
                cb = max(effective_clip_before, EVENT_CONTEXT_AI_BEFORE_SECONDS)
                ca = max(effective_clip_after, EVENT_CONTEXT_AI_AFTER_SECONDS)
                if cue_priority >= 0.86:
                    cb = max(cb, effective_clip_before + 3.0)
                    ca = max(ca, effective_clip_after + 4.0)
                elif cue_priority >= 0.72:
                    cb = max(cb, effective_clip_before + 1.5)
                    ca = max(ca, effective_clip_after + 2.5)
            seg = _build_segment_from_center(
                cue_time,
                clip_before=cb,
                clip_after=ca,
                clamp_start=interior_start,
                clamp_end=interior_end,
            )
            if seg is None:
                continue
            forced_centers.append(cue_time)
            forced_segments.append(seg)
            if len(forced_centers) >= middle_clip_target:
                break

        middle_segments.extend(forced_segments)
        remaining_slots = max(0, middle_clip_target - len(forced_centers))
        selected_centers: list[float] = []
        occupied = forced_centers[:]

        for event in ranked_events:
            if len(selected_centers) >= remaining_slots:
                break
            if event <= interior_start or event >= interior_end:
                continue
            if any(abs(event - c) < spacing_seconds for c in occupied):
                continue
            selected_centers.append(event)
            occupied.append(event)

        if len(selected_centers) < remaining_slots:
            used_fallback = True
            missing = remaining_slots - len(selected_centers)
            selected_centers.extend(
                _generate_fallback_centers(
                    start=interior_start,
                    end=interior_end,
                    count=missing,
                    min_gap_seconds=spacing_seconds,
                    existing=occupied + selected_centers,
                )
            )

        for center in sorted(selected_centers):
            seg = _build_segment_from_center(
                center,
                clip_before=effective_clip_before,
                clip_after=effective_clip_after,
                clamp_start=interior_start,
                clamp_end=interior_end,
            )
            if seg is not None:
                middle_segments.append(seg)

    all_segments = anchored_segments + middle_segments
    merge_gap = max(0.25, min_gap_seconds * (0.35 if vision_windows else 0.15))
    merged = merge_segments(all_segments, merge_gap=merge_gap)

    # ── Coverage band enforcement ────────────────────────────────────────
    coverage_bands_added: list[str] = []
    if ensure_game_phase_coverage and vision_windows and interior_end > interior_start:
        merged, coverage_bands_added = _ensure_coverage_bands(
            merged,
            interior_start=interior_start,
            interior_end=interior_end,
            vision_windows=vision_windows,
            clip_before=clip_before,
            clip_after=clip_after,
        )
        if coverage_bands_added:
            merged = merge_segments(merged, merge_gap=merge_gap)

    return merged, used_fallback, coverage_bands_added


# ---------------------------------------------------------------------------
# Adaptive cue threshold helpers
# ---------------------------------------------------------------------------


def _adaptive_ai_cue_thresholds(
    requested: float,
    *,
    minimum: float = ADAPTIVE_AI_CUE_THRESHOLD_MIN,
) -> list[float]:
    clamped = min(1.0, max(0.0, requested))
    thresholds: list[float] = []
    for factor in ADAPTIVE_AI_CUE_THRESHOLD_FACTORS:
        candidate = max(minimum, clamped * factor)
        if not any(abs(candidate - t) < 1e-6 for t in thresholds):
            thresholds.append(candidate)
    if minimum not in thresholds:
        thresholds.append(minimum)
    thresholds.sort(reverse=True)
    return thresholds


# ---------------------------------------------------------------------------
# Plan scoring (used by auto-optimize to skip N renders)
# ---------------------------------------------------------------------------


def score_plan_analytically(
    segments: list[Segment],
    *,
    vision_windows: list[VisionWindow],
    events: list[float],
    duration_seconds: float,
    ai_cues: list[TranscriptionCue] | None = None,
) -> dict[str, object]:
    """Score a segment plan using already-computed analysis signals.

    Returns a dict with the same top-level keys as :func:`analyze_watchability`
    (``watchability_score``, ``highlight_quality_score``, ``youtube_score``).
    This avoids rendering N candidate MP4s during ``auto --auto-optimize``.
    """
    if not segments or duration_seconds <= 0:
        return {
            "watchability_score": 0.0,
            "highlight_quality_score": 0.0,
            "youtube_score": 0.0,
            "_source": "analytical",
        }

    total_duration = sum(s.duration for s in segments)

    # ── Vision score of selected windows ────────────────────────────────
    window_scores: list[float] = []
    for seg in segments:
        overlapping = [w for w in vision_windows if w.end > seg.start and w.start < seg.end]
        if overlapping:
            total_overlap = sum(
                min(w.end, seg.end) - max(w.start, seg.start) for w in overlapping
            )
            if total_overlap > 0:
                ws = sum(
                    w.score * (min(w.end, seg.end) - max(w.start, seg.start))
                    for w in overlapping
                ) / total_overlap
                window_scores.append(ws)
        else:
            window_scores.append(0.08)

    avg_vision_score = _average(window_scores) if window_scores else 0.0

    # ── Scene event density ──────────────────────────────────────────────
    total_events_in_segs = sum(
        1 for e in events if any(s.start <= e <= s.end for s in segments)
    )
    events_per_minute = (
        total_events_in_segs / (total_duration / 60.0) if total_duration > 0 else 0.0
    )
    event_density_score = min(1.0, events_per_minute / 8.0)

    # ── Game-phase coverage (early / mid / late) ─────────────────────────
    band_size = duration_seconds / 3.0
    bands_covered = sum(
        1
        for band in range(3)
        if any(
            s.start < (band + 1) * band_size and s.end > band * band_size
            for s in segments
        )
    )
    coverage_score = bands_covered / 3.0

    # ── Pacing ───────────────────────────────────────────────────────────
    centers = [(s.start + s.end) / 2 for s in segments]
    if len(centers) > 1:
        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        avg_gap = _average(gaps)
        gap_cv = _stddev(gaps) / avg_gap if avg_gap > 0 else 0.0
        pacing_score = max(0.0, 1.0 - gap_cv * 0.3)
    else:
        pacing_score = 0.5

    # ── Duration ratio ────────────────────────────────────────────────────
    duration_ratio = total_duration / duration_seconds
    duration_score = _closeness_score(duration_ratio, target=0.667, tolerance=0.2)

    # ── AI cue coverage ──────────────────────────────────────────────────
    ai_cue_score = 0.5
    if ai_cues:
        covered = sum(
            1 for cue in ai_cues if any(s.start <= cue.center <= s.end for s in segments)
        )
        ai_cue_score = covered / max(1, len(ai_cues))

    # ── Blended scores (0–100) ────────────────────────────────────────────
    watchability = (
        0.35 * avg_vision_score
        + 0.25 * event_density_score
        + 0.20 * pacing_score
        + 0.20 * coverage_score
    ) * 100

    quality = (
        0.40 * avg_vision_score
        + 0.25 * duration_score
        + 0.20 * coverage_score
        + 0.15 * ai_cue_score
    ) * 100

    youtube = 0.6 * watchability + 0.4 * quality

    return {
        "watchability_score": round(watchability, 1),
        "highlight_quality_score": round(quality, 1),
        "youtube_score": round(youtube, 1),
        "coverage_bands": bands_covered,
        "avg_vision_score": round(avg_vision_score, 3),
        "event_density_per_minute": round(events_per_minute, 2),
        "pacing_score": round(pacing_score, 3),
        "duration_ratio": round(duration_ratio, 3),
        "segment_count": len(segments),
        "total_duration_seconds": round(total_duration, 1),
        "_source": "analytical",
    }


# ---------------------------------------------------------------------------
# Plan I/O
# ---------------------------------------------------------------------------


def _format_seconds(seconds: float) -> str:
    total = int(max(0, round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def write_plan(
    *,
    input_path: Path,
    output_path: Path,
    duration_seconds: float,
    events: list[float],
    segments: list[Segment],
    used_fallback: bool,
    settings: dict[str, object],
    ai_cues: list[TranscriptionCue] | None = None,
    coverage_bands_added: list[str] | None = None,
) -> None:
    payload: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "input": str(input_path),
        "duration_seconds": duration_seconds,
        "events": events,
        "used_fallback_events": used_fallback,
        "settings": settings,
        "segments": [
            {
                **asdict(segment),
                "duration": segment.duration,
                "label": f"{_format_seconds(segment.start)} - {_format_seconds(segment.end)}",
            }
            for segment in segments
        ],
    }
    if coverage_bands_added:
        payload["coverage_bands_added"] = coverage_bands_added
    if ai_cues:
        payload["ai_cues"] = [
            {
                "start": round(c.start, 3),
                "end": round(c.end, 3),
                "center": round(c.center, 3),
                "score": round(c.score, 4),
                "keywords": list(c.keywords),
                "text": c.text[:200],
            }
            for c in ai_cues
        ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rendered = json.dumps(payload, indent=2, allow_nan=False)
    except ValueError as err:
        raise EditorError("Plan contains non-finite numeric values.") from err
    output_path.write_text(rendered, encoding="utf-8")


def read_plan(plan_path: Path) -> list[Segment]:
    try:
        raw = plan_path.read_text(encoding="utf-8")
    except OSError as err:
        raise EditorError(f"Could not read plan file: {plan_path}") from err
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as err:
        raise EditorError(f"Plan file is not valid JSON: {plan_path}") from err
    if not isinstance(data, dict):
        raise EditorError(f"Plan JSON must be an object: {plan_path}")
    raw_segments = data.get("segments", [])
    if not isinstance(raw_segments, list):
        raise EditorError(f"'segments' must be a list in plan: {plan_path}")
    segments: list[Segment] = []
    for index, raw_seg in enumerate(raw_segments):
        if not isinstance(raw_seg, dict):
            raise EditorError(f"Segment {index} is not an object in plan: {plan_path}")
        try:
            start = float(raw_seg["start"])
            end = float(raw_seg["end"])
        except (KeyError, TypeError, ValueError) as err:
            raise EditorError(
                f"Segment at index {index} must have numeric 'start'/'end': {plan_path}"
            ) from err
        if not (math.isfinite(start) and math.isfinite(end)):
            raise EditorError(f"Segment {index} has non-finite start/end: {plan_path}")
        if start < 0 or end < 0:
            raise EditorError(f"Segment {index} has negative start/end: {plan_path}")
        if end <= start:
            raise EditorError(f"Segment {index} must have end > start: {plan_path}")
        segments.append(Segment(start=start, end=end))
    return segments


# ---------------------------------------------------------------------------
# Auto-optimize variant generation
# ---------------------------------------------------------------------------


def build_auto_optimize_variants(
    *,
    scene_threshold: float,
    clip_before: float,
    clip_after: float,
    min_gap_seconds: float,
    max_clips: int,
    target_duration_ratio: float,
    ai_cue_threshold: float,
    ocr_cue_threshold: float,
    candidate_count: int,
) -> list[dict[str, float | int]]:
    templates: list[dict[str, float | int]] = [
        {
            "scene_threshold": scene_threshold,
            "clip_before": clip_before,
            "clip_after": clip_after,
            "min_gap_seconds": min_gap_seconds,
            "max_clips": max_clips,
            "target_duration_ratio": target_duration_ratio,
            "ai_cue_threshold": ai_cue_threshold,
            "ocr_cue_threshold": ocr_cue_threshold,
        },
        {
            "scene_threshold": scene_threshold - 0.02,
            "clip_before": clip_before + 2.0,
            "clip_after": clip_after + 3.0,
            "min_gap_seconds": min_gap_seconds - 1.0,
            "max_clips": max_clips + 3,
            "target_duration_ratio": target_duration_ratio + 0.03,
            "ai_cue_threshold": ai_cue_threshold - 0.04,
            "ocr_cue_threshold": ocr_cue_threshold - 0.02,
        },
        {
            "scene_threshold": scene_threshold - 0.04,
            "clip_before": clip_before - 1.0,
            "clip_after": clip_after - 2.0,
            "min_gap_seconds": min_gap_seconds - 2.0,
            "max_clips": max_clips + 6,
            "target_duration_ratio": target_duration_ratio - 0.06,
            "ai_cue_threshold": ai_cue_threshold - 0.06,
            "ocr_cue_threshold": ocr_cue_threshold - 0.03,
        },
        {
            "scene_threshold": scene_threshold + 0.02,
            "clip_before": clip_before + 1.5,
            "clip_after": clip_after + 1.0,
            "min_gap_seconds": min_gap_seconds + 1.0,
            "max_clips": max_clips + 1,
            "target_duration_ratio": target_duration_ratio + 0.02,
            "ai_cue_threshold": ai_cue_threshold,
            "ocr_cue_threshold": ocr_cue_threshold,
        },
        {
            "scene_threshold": scene_threshold - 0.01,
            "clip_before": clip_before + 3.0,
            "clip_after": clip_after + 4.0,
            "min_gap_seconds": min_gap_seconds - 1.0,
            "max_clips": max_clips + 4,
            "target_duration_ratio": target_duration_ratio + 0.05,
            "ai_cue_threshold": ai_cue_threshold - 0.03,
            "ocr_cue_threshold": ocr_cue_threshold - 0.02,
        },
        {
            "scene_threshold": scene_threshold + 0.01,
            "clip_before": clip_before - 0.5,
            "clip_after": clip_after - 1.0,
            "min_gap_seconds": min_gap_seconds + 2.0,
            "max_clips": max_clips - 2,
            "target_duration_ratio": target_duration_ratio - 0.04,
            "ai_cue_threshold": ai_cue_threshold + 0.02,
            "ocr_cue_threshold": ocr_cue_threshold + 0.01,
        },
        {
            "scene_threshold": scene_threshold - 0.03,
            "clip_before": clip_before + 1.0,
            "clip_after": clip_after + 2.0,
            "min_gap_seconds": min_gap_seconds,
            "max_clips": max_clips + 5,
            "target_duration_ratio": target_duration_ratio - 0.02,
            "ai_cue_threshold": ai_cue_threshold - 0.05,
            "ocr_cue_threshold": ocr_cue_threshold - 0.04,
        },
        {
            "scene_threshold": scene_threshold,
            "clip_before": clip_before + 0.5,
            "clip_after": clip_after + 1.5,
            "min_gap_seconds": min_gap_seconds - 1.0,
            "max_clips": max_clips + 2,
            "target_duration_ratio": target_duration_ratio,
            "ai_cue_threshold": ai_cue_threshold - 0.02,
            "ocr_cue_threshold": ocr_cue_threshold,
        },
    ]
    deduped: list[dict[str, float | int]] = []
    seen: set[tuple[float, ...]] = set()
    for template in templates:
        normalized: dict[str, float | int] = {
            "scene_threshold": max(0.08, min(0.60, float(template["scene_threshold"]))),
            "clip_before": max(3.0, min(26.0, float(template["clip_before"]))),
            "clip_after": max(5.0, min(34.0, float(template["clip_after"]))),
            "min_gap_seconds": max(6.0, min(28.0, float(template["min_gap_seconds"]))),
            "max_clips": int(max(14, min(44, int(round(float(template["max_clips"])))))),
            "target_duration_ratio": max(0.20, min(0.75, float(template["target_duration_ratio"]))),
            "ai_cue_threshold": max(0.08, min(0.80, float(template["ai_cue_threshold"]))),
            "ocr_cue_threshold": max(0.08, min(0.70, float(template["ocr_cue_threshold"]))),
        }
        sig = (
            round(float(normalized["scene_threshold"]), 4),
            round(float(normalized["clip_before"]), 3),
            round(float(normalized["clip_after"]), 3),
            round(float(normalized["min_gap_seconds"]), 3),
            float(normalized["max_clips"]),
            round(float(normalized["target_duration_ratio"]), 4),
            round(float(normalized["ai_cue_threshold"]), 4),
            round(float(normalized["ocr_cue_threshold"]), 4),
        )
        if sig in seen:
            continue
        seen.add(sig)
        deduped.append(normalized)
        if len(deduped) >= candidate_count:
            break
    return deduped


def select_optimize_metric_value(
    *,
    report: dict[str, object],
    metric: str,
) -> float:
    key = {"youtube": "youtube_score", "watchability": "watchability_score", "quality": "highlight_quality_score"}[metric]
    value = report.get(key, 0.0)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Main analysis entry point (with cache integration)
# ---------------------------------------------------------------------------


def analyze_recording(
    *,
    input_path: Path,
    plan_path: Path,
    scene_threshold: float,
    clip_before: float,
    clip_after: float,
    min_gap_seconds: float,
    max_clips: int,
    target_duration_seconds: float,
    target_duration_ratio: float,
    intro_seconds: float,
    outro_seconds: float,
    vision_scoring: str,
    vision_sample_fps: float,
    vision_window_seconds: float,
    vision_step_seconds: float,
    whisper_model: Path | None,
    whisper_bin: str,
    whisper_language: str,
    whisper_threads: int,
    whisper_audio_stream: int,
    whisper_vad: bool,
    whisper_vad_threshold: float,
    whisper_vad_model: Path | None,
    ocr_cue_scoring: str,
    tesseract_bin: str,
    ocr_sample_fps: float,
    ocr_cue_threshold: float,
    ai_cue_threshold: float,
    end_on_result: bool = True,
    result_detect_fps: float = DEFAULT_RESULT_DETECT_FPS,
    result_detect_tail_seconds: float = DEFAULT_RESULT_DETECT_TAIL_SECONDS,
    one_shot_smart: bool = DEFAULT_ONE_SHOT_SMART,
    cache: CacheStore | None = None,
) -> dict[str, object]:
    """Analyse a recording and write an edit plan.

    All expensive ffmpeg passes are transparently cached when *cache* is
    provided.  A second run on the same file with the same settings returns
    almost instantly.

    Returns a stats dict (segment_count, event_count, etc.).
    """
    from .ffmpeg_utils import probe_duration_seconds  # avoid circular at module level

    source_duration_seconds = probe_duration_seconds(input_path)
    duration_seconds = source_duration_seconds
    terminal_result_seconds: float | None = None

    # ── End-of-match OCR (cached) ─────────────────────────────────────────
    if end_on_result:
        try:
            if cache is not None:
                r_key = result_time_key(
                    input_path, sample_fps=result_detect_fps, tail_seconds=result_detect_tail_seconds
                )
                cached_result = cache.get(r_key, "result_time")
                if cached_result is not None:
                    terminal_result_seconds = (
                        None if cached_result.get("value") is None else float(cached_result["value"])
                    )
                    print("Using cached end-of-match detection.", file=sys.stderr)
                else:
                    terminal_result_seconds = _detect_terminal_result_time(
                        input_path=input_path,
                        duration_seconds=source_duration_seconds,
                        tesseract_binary=tesseract_bin,
                        sample_fps=result_detect_fps,
                        tail_seconds=result_detect_tail_seconds,
                    )
                    cache.put(r_key, "result_time", {"value": terminal_result_seconds})
            else:
                terminal_result_seconds = _detect_terminal_result_time(
                    input_path=input_path,
                    duration_seconds=source_duration_seconds,
                    tesseract_binary=tesseract_bin,
                    sample_fps=result_detect_fps,
                    tail_seconds=result_detect_tail_seconds,
                )
        except (EditorError, subprocess.CalledProcessError) as error:
            print(
                f"Warning: end-of-match detection failed ({error}); using full recording duration.",
                file=sys.stderr,
                flush=True,
            )
        if terminal_result_seconds is not None:
            duration_seconds = max(1.0, min(source_duration_seconds, terminal_result_seconds))
            print(
                f"Detected match end at {duration_seconds:.1f}s via Victory/Defeat OCR.",
                file=sys.stderr,
                flush=True,
            )

    # ── Scene event detection (cached) ────────────────────────────────────
    events: list[float]
    scene_threshold_used: float
    if cache is not None:
        sc_key = scene_events_key(input_path, scene_threshold)
        cached_events = cache.get(sc_key, "scene_events")
        if cached_events is not None:
            events = [float(t) for t in cached_events.get("events", [])]
            scene_threshold_used = float(cached_events.get("threshold_used", scene_threshold))
            print(
                f"Using cached scene events ({len(events)} events at threshold {scene_threshold_used:.3f}).",
                file=sys.stderr,
            )
        else:
            events, scene_threshold_used = detect_scene_events_adaptive(
                input_path,
                scene_threshold,
                duration_seconds=duration_seconds,
                show_progress=True,
            )
            cache.put(
                sc_key,
                "scene_events",
                {"events": events, "threshold_used": scene_threshold_used},
            )
    else:
        events, scene_threshold_used = detect_scene_events_adaptive(
            input_path,
            scene_threshold,
            duration_seconds=duration_seconds,
            show_progress=True,
        )
    events = [e for e in events if 0.0 <= e <= duration_seconds]

    resolved_target = target_duration_seconds
    if resolved_target <= 0 and target_duration_ratio > 0:
        resolved_target = duration_seconds * min(1.0, target_duration_ratio)

    # ── Vision scoring (cached) ───────────────────────────────────────────
    vision_windows: list[VisionWindow] = []
    if vision_scoring != "off":
        # Cropdetect first (cached separately — independent of sample_fps)
        crop_filter: str | None = None
        if cache is not None:
            cd_key = cropdetect_key(input_path)
            cached_crop = cache.get(cd_key, "cropdetect")
            if cached_crop is not None:
                crop_filter = cached_crop.get("crop_filter")
            else:
                crop_filter = detect_crop_filter(input_path, duration_seconds=duration_seconds)
                cache.put(cd_key, "cropdetect", {"crop_filter": crop_filter})
        else:
            crop_filter = detect_crop_filter(input_path, duration_seconds=duration_seconds)

        # Signalstats samples (cached)
        if cache is not None:
            ss_key = signalstats_key(input_path, vision_sample_fps)
            cached_ss = cache.get(ss_key, "signalstats")
            if cached_ss is not None:
                raw_samples = cached_ss.get("samples", [])
                samples: list[tuple[float, float, float]] = [
                    (float(s[0]), float(s[1]), float(s[2])) for s in raw_samples
                ]
                print(
                    f"Using cached signalstats ({len(samples)} samples).",
                    file=sys.stderr,
                )
            else:
                samples = extract_signalstats_samples(
                    input_path,
                    duration_seconds=duration_seconds,
                    sample_fps=vision_sample_fps,
                    crop_filter=crop_filter,
                    show_progress=True,
                )
                cache.put(ss_key, "signalstats", {"samples": [list(s) for s in samples]})
        else:
            samples = extract_signalstats_samples(
                input_path,
                duration_seconds=duration_seconds,
                sample_fps=vision_sample_fps,
                crop_filter=crop_filter,
                show_progress=True,
            )

        try:
            vision_windows = _compute_vision_window_scores(
                frame_samples=samples,
                events=events,
                duration_seconds=duration_seconds,
                window_seconds=vision_window_seconds,
                step_seconds=vision_step_seconds,
            )
        except subprocess.CalledProcessError as error:
            print("Warning: vision scoring failed; falling back to scene-only.", file=sys.stderr)
            if error.stderr:
                print(error.stderr, file=sys.stderr)
        vision_windows = [w for w in vision_windows if w.end <= duration_seconds]

    # ── Local AI (whisper + OCR) ───────────────────────────────────────────
    ai_cues: list[TranscriptionCue] = []
    whisper_cues: list[TranscriptionCue] = []
    ocr_cues: list[TranscriptionCue] = []
    ai_cue_threshold_used = ai_cue_threshold

    if vision_scoring == "local-ai":
        if whisper_model is None:
            raise EditorError("--whisper-model is required when --vision-scoring local-ai.")
        print("Running local AI transcription (whisper.cpp)...", file=sys.stderr, flush=True)
        resolved_whisper_vad = whisper_vad and whisper_vad_model is not None
        if whisper_vad and whisper_vad_model is None:
            print(
                "Warning: --whisper-vad requested but no --whisper-vad-model; continuing without VAD.",
                file=sys.stderr,
                flush=True,
            )

        # Whisper cues (cached)
        w_key = (
            whisper_key(
                input_path,
                model_path=whisper_model,
                language=whisper_language,
                audio_stream=whisper_audio_stream if whisper_audio_stream >= 0 else None,
                vad=resolved_whisper_vad,
                vad_threshold=whisper_vad_threshold,
            )
            if cache is not None
            else None
        )
        raw_whisper_cues: list[TranscriptionCue] | None = None
        if cache is not None and w_key is not None:
            cached_w = cache.get(w_key, "whisper")
            if cached_w is not None:
                raw_whisper_cues = [
                    TranscriptionCue(
                        start=float(c["start"]),
                        end=float(c["end"]),
                        score=float(c["score"]),
                        text=str(c["text"]),
                        keywords=tuple(c.get("keywords", [])),
                    )
                    for c in cached_w.get("cues", [])
                ]
                print(
                    f"Using cached whisper cues ({len(raw_whisper_cues)} cues).",
                    file=sys.stderr,
                )

        if raw_whisper_cues is None:
            try:
                raw_whisper_cues = _collect_local_ai_cues(
                    input_path=input_path,
                    whisper_model=whisper_model,
                    whisper_binary=whisper_bin,
                    whisper_language=whisper_language,
                    whisper_threads=whisper_threads,
                    cue_threshold=ADAPTIVE_AI_CUE_THRESHOLD_MIN,
                    whisper_audio_stream=(
                        None if whisper_audio_stream < 0 else whisper_audio_stream
                    ),
                    whisper_vad=resolved_whisper_vad,
                    whisper_vad_threshold=whisper_vad_threshold,
                    whisper_vad_model=whisper_vad_model,
                )
            except subprocess.CalledProcessError as error:
                detail = f"\n{(error.stderr or '').strip()}" if error.stderr else ""
                raise EditorError(
                    f"Local AI transcription failed.{detail}"
                ) from error
            if cache is not None and w_key is not None:
                cache.put(
                    w_key,
                    "whisper",
                    {
                        "cues": [
                            {
                                "start": c.start,
                                "end": c.end,
                                "score": c.score,
                                "text": c.text,
                                "keywords": list(c.keywords),
                            }
                            for c in raw_whisper_cues
                        ]
                    },
                )

        raw_whisper_cues = [c for c in raw_whisper_cues if c.end <= duration_seconds]
        thresholds = _adaptive_ai_cue_thresholds(ai_cue_threshold)
        for candidate_threshold in thresholds:
            candidate_cues = [c for c in raw_whisper_cues if c.score >= candidate_threshold]
            if candidate_cues:
                whisper_cues = candidate_cues
                ai_cue_threshold_used = candidate_threshold
                break
        if whisper_cues and ai_cue_threshold_used < ai_cue_threshold:
            print(
                f"Local AI cue threshold adjusted to {ai_cue_threshold_used:.2f}; "
                f"detected {len(whisper_cues)} transcript cues.",
                file=sys.stderr,
                flush=True,
            )

        # OCR cues (cached)
        if ocr_cue_scoring != "off":
            print("Running local OCR cue detection (tesseract)...", file=sys.stderr, flush=True)
            ocr_k = (
                ocr_cues_key(input_path, ocr_sample_fps, ocr_cue_threshold)
                if cache is not None
                else None
            )
            cached_ocr_cues: list[TranscriptionCue] | None = None
            if cache is not None and ocr_k is not None:
                cached_ocr = cache.get(ocr_k, "ocr_cues")
                if cached_ocr is not None:
                    cached_ocr_cues = [
                        TranscriptionCue(
                            start=float(c["start"]),
                            end=float(c["end"]),
                            score=float(c["score"]),
                            text=str(c["text"]),
                            keywords=tuple(c.get("keywords", [])),
                        )
                        for c in cached_ocr.get("cues", [])
                    ]
                    print(
                        f"Using cached OCR cues ({len(cached_ocr_cues)} cues).",
                        file=sys.stderr,
                    )
            if cached_ocr_cues is None:
                try:
                    ocr_cues = _collect_ocr_cues(
                        input_path=input_path,
                        tesseract_binary=tesseract_bin,
                        sample_fps=ocr_sample_fps,
                        cue_threshold=ocr_cue_threshold,
                    )
                    ocr_cues = [c for c in ocr_cues if c.end <= duration_seconds]
                    if cache is not None and ocr_k is not None:
                        cache.put(
                            ocr_k,
                            "ocr_cues",
                            {
                                "cues": [
                                    {
                                        "start": c.start,
                                        "end": c.end,
                                        "score": c.score,
                                        "text": c.text,
                                        "keywords": list(c.keywords),
                                    }
                                    for c in ocr_cues
                                ]
                            },
                        )
                    if ocr_cues:
                        print(
                            f"Detected {len(ocr_cues)} OCR cues from HUD text.",
                            file=sys.stderr,
                            flush=True,
                        )
                except EditorError as error:
                    print(
                        f"Warning: OCR cue detection unavailable ({error}).",
                        file=sys.stderr,
                        flush=True,
                    )
                except subprocess.CalledProcessError as error:
                    detail = f" ({(error.stderr or '').strip()})" if error.stderr else ""
                    print(
                        f"Warning: OCR cue detection failed{detail}.",
                        file=sys.stderr,
                        flush=True,
                    )
            else:
                ocr_cues = cached_ocr_cues

        combined = sorted(whisper_cues + ocr_cues, key=lambda c: (c.start, -c.score))
        for cue in combined:
            if ai_cues and cue.start - ai_cues[-1].start < 8.0:
                if cue.score > ai_cues[-1].score:
                    ai_cues[-1] = cue
                continue
            ai_cues.append(cue)

        if not ai_cues:
            print(
                "Local AI did not detect cues; using vision/scene scoring only.",
                file=sys.stderr,
                flush=True,
            )

    ai_priority_cues = _extract_ai_priority_cues(ai_cues)
    if ai_priority_cues:
        events = sorted(events + ai_priority_cues)
        vision_windows = _boost_vision_windows_with_ai_cues(vision_windows, ai_cues)

    # ── One-shot smart tuning ─────────────────────────────────────────────
    effective_clip_before = clip_before
    effective_clip_after = clip_after
    effective_min_gap_seconds = min_gap_seconds
    effective_max_clips = max_clips
    effective_forced_cue_share = 0.60
    if one_shot_smart:
        tuned = _derive_one_shot_segment_tuning(
            events=events,
            duration_seconds=duration_seconds,
            vision_windows=vision_windows,
            ai_cues=ai_cues,
            clip_before=clip_before,
            clip_after=clip_after,
            min_gap_seconds=min_gap_seconds,
            max_clips=max_clips,
            forced_cue_share=effective_forced_cue_share,
        )
        effective_clip_before = float(tuned["clip_before"])
        effective_clip_after = float(tuned["clip_after"])
        effective_min_gap_seconds = float(tuned["min_gap_seconds"])
        effective_max_clips = int(tuned["max_clips"])
        effective_forced_cue_share = float(tuned["forced_cue_share"])
        if (
            abs(effective_clip_before - clip_before) > 1e-3
            or abs(effective_clip_after - clip_after) > 1e-3
            or abs(effective_min_gap_seconds - min_gap_seconds) > 1e-3
            or effective_max_clips != max_clips
        ):
            print(
                f"One-shot tuning: clip-before={effective_clip_before:.1f}, "
                f"clip-after={effective_clip_after:.1f}, "
                f"min-gap={effective_min_gap_seconds:.1f}, "
                f"max-clips={effective_max_clips}, "
                f"forced-cue-share={effective_forced_cue_share:.2f}",
                file=sys.stderr,
                flush=True,
            )

    # ── Build segments ────────────────────────────────────────────────────
    segments, used_fallback, coverage_bands_added = build_segments(
        events,
        duration_seconds=duration_seconds,
        clip_before=effective_clip_before,
        clip_after=effective_clip_after,
        min_gap_seconds=effective_min_gap_seconds,
        max_clips=effective_max_clips,
        target_duration_seconds=resolved_target,
        intro_seconds=intro_seconds,
        outro_seconds=outro_seconds,
        vision_windows=vision_windows,
        ai_priority_cues=ai_priority_cues,
        ai_priority_details=ai_cues,
        force_outro_to_duration_end=terminal_result_seconds is not None,
        forced_cue_share=effective_forced_cue_share,
    )
    if coverage_bands_added:
        print(
            f"Coverage bands added for: {', '.join(coverage_bands_added)}.",
            file=sys.stderr,
            flush=True,
        )

    # ── Write plan ────────────────────────────────────────────────────────
    settings: dict[str, object] = {
        "scene_threshold": scene_threshold_used,
        "scene_threshold_requested": scene_threshold,
        "clip_before": clip_before,
        "clip_after": clip_after,
        "min_gap_seconds": min_gap_seconds,
        "max_clips": max_clips,
        "effective_clip_before": effective_clip_before,
        "effective_clip_after": effective_clip_after,
        "effective_min_gap_seconds": effective_min_gap_seconds,
        "effective_max_clips": effective_max_clips,
        "effective_forced_cue_share": effective_forced_cue_share,
        "target_duration_seconds": resolved_target,
        "target_duration_seconds_requested": target_duration_seconds,
        "target_duration_ratio": target_duration_ratio,
        "intro_seconds": intro_seconds,
        "outro_seconds": outro_seconds,
        "source_duration_seconds": source_duration_seconds,
        "analysis_duration_seconds": duration_seconds,
        "end_on_result": end_on_result,
        "result_detect_fps": result_detect_fps,
        "result_detect_tail_seconds": result_detect_tail_seconds,
        "one_shot_smart": one_shot_smart,
        "terminal_result_seconds": (
            round(terminal_result_seconds, 3) if terminal_result_seconds is not None else None
        ),
        "vision_scoring": vision_scoring,
        "vision_sample_fps": vision_sample_fps,
        "vision_window_seconds": vision_window_seconds,
        "vision_step_seconds": vision_step_seconds,
        "whisper_model": str(whisper_model) if whisper_model is not None else "",
        "whisper_bin": whisper_bin,
        "whisper_language": whisper_language,
        "whisper_threads": whisper_threads,
        "whisper_audio_stream": whisper_audio_stream,
        "whisper_vad": whisper_vad,
        "whisper_vad_threshold": whisper_vad_threshold,
        "ocr_cue_scoring": ocr_cue_scoring,
        "ocr_sample_fps": ocr_sample_fps,
        "ocr_cue_threshold": ocr_cue_threshold,
        "ai_cue_threshold": ai_cue_threshold,
        "ai_cue_threshold_used": ai_cue_threshold_used,
        "coverage_bands_added": coverage_bands_added,
    }
    write_plan(
        input_path=input_path,
        output_path=plan_path,
        duration_seconds=duration_seconds,
        events=events,
        segments=segments,
        used_fallback=used_fallback,
        settings=settings,
        ai_cues=ai_cues if ai_cues else None,
        coverage_bands_added=coverage_bands_added,
    )

    return {
        "segment_count": len(segments),
        "event_count": len(events),
        "ai_cue_count": len(ai_cues),
        "whisper_cue_count": len(whisper_cues),
        "ocr_cue_count": len(ocr_cues),
        "used_fallback": used_fallback,
        "scene_threshold_used": scene_threshold_used,
        "coverage_bands_added": coverage_bands_added,
        # expose vision_windows and events for plan scoring (auto-optimize)
        "_vision_windows": vision_windows,
        "_events": events,
        "_ai_cues": ai_cues,
        "_duration_seconds": duration_seconds,
    }
