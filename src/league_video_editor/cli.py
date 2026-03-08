from __future__ import annotations

import argparse
from collections.abc import Callable
from collections import Counter, deque
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


SCENE_PTS_PATTERN = re.compile(r"pts_time:(-?\d+(?:\.\d+)?)")
FFMPEG_TIME_PATTERN = re.compile(r"time=(\d{2}:\d{2}:\d{2}(?:\.\d+)?)")
SIGNALSTATS_FRAME_PATTERN = re.compile(r"frame:\s*\d+\s+pts:\s*-?\d+\s+pts_time:(-?\d+(?:\.\d+)?)")
SIGNALSTATS_YDIF_PATTERN = re.compile(r"lavfi\.signalstats\.YDIF=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
SIGNALSTATS_SATAVG_PATTERN = re.compile(
    r"lavfi\.signalstats\.SATAVG=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
CROPDETECT_PATTERN = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")
WHISPER_TIMESTAMP_PATTERN = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})(?:[.,](\d{1,3}))?$")
LOUDNORM_FILTER = "loudnorm=I=-14:LRA=11:TP=-1.5"
LOUDNORM_NONFINITE_PATTERN = re.compile(r"input contains.*nan.*inf", re.IGNORECASE | re.DOTALL)
LOUDNORM_ANALYSIS_FILTER = f"{LOUDNORM_FILTER}:print_format=json"
VIDEO_ENCODERS = ("libx264", "h264_videotoolbox", "h264_nvenc", "h264_qsv", "h264_amf")
DEFAULT_TARGET_DURATION_SECONDS = 1200.0
DEFAULT_TARGET_DURATION_RATIO = 2.0 / 3.0
DEFAULT_INTRO_SECONDS = 120.0
DEFAULT_OUTRO_SECONDS = 60.0
DEFAULT_VISION_SCORING = "heuristic"
DEFAULT_VISION_SAMPLE_FPS = 1.0
DEFAULT_VISION_WINDOW_SECONDS = 12.0
DEFAULT_VISION_STEP_SECONDS = 6.0
DEFAULT_AI_CUE_THRESHOLD = 0.40
DEFAULT_WHISPER_CPP_BIN = "auto"
DEFAULT_WHISPER_LANGUAGE = "en"
DEFAULT_WHISPER_THREADS = max(1, min(8, os.cpu_count() or 4))
DEFAULT_WHISPER_VAD = True
DEFAULT_WHISPER_VAD_THRESHOLD = 0.50
DEFAULT_TESSERACT_BIN = "auto"
DEFAULT_OCR_CUE_SCORING = "auto"
DEFAULT_OCR_SAMPLE_FPS = 0.25
DEFAULT_OCR_CUE_THRESHOLD = 0.16
DEFAULT_RESULT_DETECT_FPS = 0.33
DEFAULT_RESULT_DETECT_TAIL_SECONDS = 900.0
DEFAULT_AUTO_OPTIMIZE_CANDIDATES = 6
DEFAULT_AUTO_OPTIMIZE_METRIC = "youtube"
DEFAULT_ONE_SHOT_SMART = True
DEFAULT_THUMBNAIL_WIDTH = 1280
DEFAULT_THUMBNAIL_HEIGHT = 720
DEFAULT_THUMBNAIL_QUALITY = 2
DEFAULT_THUMBNAIL_SCENE_THRESHOLD = 0.20
DEFAULT_THUMBNAIL_VISION_SAMPLE_FPS = 0.75
DEFAULT_THUMBNAIL_VISION_WINDOW_SECONDS = 8.0
DEFAULT_THUMBNAIL_VISION_STEP_SECONDS = 4.0
DEFAULT_THUMBNAIL_CHAMPION_SCALE = 0.55
DEFAULT_THUMBNAIL_CHAMPION_ANCHOR = "right"
DEFAULT_THUMBNAIL_HEADLINE_SIZE = 118
DEFAULT_THUMBNAIL_HEADLINE_COLOR = "white"
DEFAULT_THUMBNAIL_HEADLINE_Y_RATIO = 0.06
DEFAULT_DESCRIPTION_OUTPUT = "youtube-description.txt"
DEFAULT_DESCRIPTION_TITLE_COUNT = 3
DEFAULT_DESCRIPTION_MAX_MOMENTS = 8
ADAPTIVE_AI_CUE_THRESHOLD_MIN = 0.12
ADAPTIVE_AI_CUE_THRESHOLD_FACTORS = (1.0, 0.82, 0.65, 0.48)
ADAPTIVE_SCENE_THRESHOLD_MIN = 0.08
ADAPTIVE_SCENE_THRESHOLD_FACTORS = (1.0, 0.75, 0.6, 0.45)
EVENT_CONTEXT_DEATH_BEFORE_SECONDS = 18.0
EVENT_CONTEXT_DEATH_AFTER_SECONDS = 9.0
EVENT_CONTEXT_COMBAT_BEFORE_SECONDS = 12.0
EVENT_CONTEXT_COMBAT_AFTER_SECONDS = 8.0
EVENT_CONTEXT_AI_BEFORE_SECONDS = 14.0
EVENT_CONTEXT_AI_AFTER_SECONDS = 9.0
AI_WINDOW_BOOST_RADIUS_SECONDS = 22.0

TRANSCRIPT_HYPE_WEIGHTS = {
    "pentakill": 1.00,
    "quadra kill": 0.90,
    "triple kill": 0.82,
    "double kill": 0.66,
    "killing spree": 0.70,
    "rampage": 0.76,
    "unstoppable": 0.72,
    "godlike": 0.82,
    "legendary": 0.85,
    "slain": 0.55,
    "first blood": 0.78,
    "enemy slain": 0.70,
    "ally slain": 0.52,
    "shutdown": 0.82,
    "ace": 0.78,
    "baron": 0.80,
    "baron nashor": 0.85,
    "elder dragon": 0.85,
    "dragon": 0.52,
    "rift herald": 0.60,
    "nashor": 0.74,
    "turret": 0.44,
    "tower": 0.42,
    "structure destroyed": 0.58,
    "inhibitor": 0.48,
    "inhib": 0.42,
    "nexus": 0.96,
    "annihilated": 0.62,
    "executed": 0.48,
    "victory": 0.64,
    "defeat": 0.50,
    "teamfight": 0.72,
    "fight": 0.44,
    "outplay": 0.68,
    "gank": 0.62,
    "dive": 0.54,
    "stolen": 0.64,
    "smite": 0.42,
    "kill": 0.40,
    "clutch": 0.62,
    "flash": 0.34,
    "ultimate": 0.32,
    "ult": 0.30,
}


def _compile_hype_phrase_pattern(phrase: str) -> re.Pattern[str]:
    parts = [re.escape(piece) for piece in phrase.split() if piece]
    if not parts:
        return re.compile(r"^\b\B$")
    if len(parts) == 1:
        return re.compile(rf"\b{parts[0]}\b")
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b")


TRANSCRIPT_HYPE_PATTERNS = tuple(
    (phrase, weight, _compile_hype_phrase_pattern(phrase))
    for phrase, weight in TRANSCRIPT_HYPE_WEIGHTS.items()
)


class EditorError(RuntimeError):
    """Raised when the editor cannot proceed."""


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class VisionWindow:
    start: float
    end: float
    score: float
    motion: float
    saturation: float
    scene_density: float


@dataclass(frozen=True)
class TranscriptionCue:
    start: float
    end: float
    score: float
    text: str
    keywords: tuple[str, ...]

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0


def _require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise EditorError(
            f"'{name}' is required but not installed. Install ffmpeg/ffprobe first."
        )


def _run_command(cmd: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def _video_postprocess_filter(
    *,
    allow_upscale: bool,
    crop_filter: str | None = None,
) -> str:
    if allow_upscale:
        scale = "scale=w=1920:h=1080:force_original_aspect_ratio=decrease"
    else:
        scale = (
            "scale="
            "w='if(gt(iw,1920),1920,iw)':"
            "h='if(gt(ih,1080),1080,ih)':"
            "force_original_aspect_ratio=decrease"
        )
    pieces: list[str] = []
    if crop_filter:
        pieces.append(crop_filter)
    pieces.extend(
        [
            scale,
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black",
            "fps=60",
            "format=yuv420p",
        ]
    )
    return ",".join(pieces)


def _video_codec_args(*, video_encoder: str, crf: int, preset: str) -> list[str]:
    if video_encoder == "libx264":
        return [
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
        ]
    return [
        "-c:v",
        video_encoder,
        "-b:v",
        "10M",
        "-maxrate",
        "12M",
        "-bufsize",
        "20M",
        "-pix_fmt",
        "yuv420p",
    ]


def _effective_transition_duration(
    requested: float,
    *,
    left_duration: float,
    right_duration: float,
) -> float:
    if requested <= 0:
        return 0.0
    max_duration = min(left_duration, right_duration) - 0.001
    if max_duration <= 0:
        raise EditorError(
            "Crossfade requires segment durations greater than 0.001 seconds."
        )
    return min(requested, max_duration)


def _probe_duration_seconds(input_path: Path) -> float:
    result = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ],
        capture_output=True,
    )
    duration_text = (result.stdout or "").strip()
    if not duration_text:
        raise EditorError(f"Could not read duration from {input_path}")
    try:
        duration = float(duration_text)
    except ValueError as error:
        raise EditorError(f"Could not parse duration from {input_path}: {duration_text!r}") from error
    if not math.isfinite(duration) or duration < 0:
        raise EditorError(f"Duration is not a valid finite value for {input_path}: {duration_text!r}")
    return duration


def _parse_ffmpeg_clock_to_seconds(clock: str) -> float:
    parts = clock.split(":")
    if len(parts) != 3:
        return 0.0
    try:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    except ValueError:
        return 0.0
    return max(0.0, hours * 3600 + minutes * 60 + seconds)


def _render_progress_line(label: str, ratio: float, *, width: int = 28) -> str:
    clamped = min(1.0, max(0.0, ratio))
    filled = int(clamped * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"\r{label}: [{bar}] {clamped * 100:5.1f}%"


def _detect_scene_events_with_progress(
    cmd: list[str],
    *,
    duration_seconds: float,
    progress_label: str | None = "Analyzing scenes",
    progress_callback: Callable[[float], None] | None = None,
) -> list[float]:
    events: set[float] = set()
    last_output_ratio = -1.0
    stderr_tail: deque[str] = deque(maxlen=120)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    try:
        assert process.stderr is not None
        for line in process.stderr:
            stderr_tail.append(line)
            for value in SCENE_PTS_PATTERN.findall(line):
                try:
                    event_time = float(value)
                except ValueError:
                    continue
                if event_time >= 0:
                    events.add(event_time)

            time_match = FFMPEG_TIME_PATTERN.search(line)
            if time_match and duration_seconds > 0:
                current_seconds = _parse_ffmpeg_clock_to_seconds(time_match.group(1))
                progress_ratio = min(1.0, current_seconds / duration_seconds)
                if progress_ratio >= last_output_ratio + 0.01 or progress_ratio >= 1.0:
                    if progress_label is not None:
                        print(
                            _render_progress_line(progress_label, progress_ratio),
                            end="",
                            file=sys.stderr,
                            flush=True,
                        )
                    if progress_callback is not None:
                        progress_callback(progress_ratio)
                    last_output_ratio = progress_ratio

        return_code = process.wait()
        if duration_seconds > 0:
            if progress_label is not None:
                print(
                    _render_progress_line(progress_label, 1.0),
                    file=sys.stderr,
                    flush=True,
                )
            if progress_callback is not None:
                progress_callback(1.0)
        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code,
                cmd,
                stderr="".join(stderr_tail),
            )
        return sorted(events)
    except KeyboardInterrupt:
        process.terminate()
        raise


def _has_audio_stream(input_path: Path) -> bool:
    result = _run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(input_path),
        ],
        capture_output=True,
    )
    return bool(result.stdout.strip())


def detect_scene_events(
    input_path: Path,
    scene_threshold: float,
    *,
    duration_seconds: float | None = None,
    show_progress: bool = False,
    progress_label: str | None = "Analyzing scenes",
    progress_callback: Callable[[float], None] | None = None,
) -> list[float]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-stats_period",
        "0.25",
        "-i",
        str(input_path),
        "-filter:v",
        f"select='gt(scene,{scene_threshold})',showinfo",
        "-an",
        "-f",
        "null",
        "-",
    ]
    if show_progress:
        if duration_seconds is None:
            duration_seconds = _probe_duration_seconds(input_path)
        return _detect_scene_events_with_progress(
            cmd,
            duration_seconds=duration_seconds,
            progress_label=progress_label,
            progress_callback=progress_callback,
        )

    result = _run_command(cmd, capture_output=True)
    output = (result.stderr or "") + "\n" + (result.stdout or "")
    matches = SCENE_PTS_PATTERN.findall(output)
    return sorted({float(value) for value in matches if float(value) >= 0.0})


def _adaptive_scene_thresholds(
    scene_threshold: float,
    *,
    minimum_threshold: float = ADAPTIVE_SCENE_THRESHOLD_MIN,
) -> list[float]:
    clamped_threshold = min(1.0, max(0.0, scene_threshold))
    thresholds: list[float] = []
    for factor in ADAPTIVE_SCENE_THRESHOLD_FACTORS:
        candidate = max(minimum_threshold, clamped_threshold * factor)
        if any(abs(candidate - existing) < 1e-6 for existing in thresholds):
            continue
        thresholds.append(candidate)
    if not thresholds:
        thresholds.append(clamped_threshold)
    return thresholds


def detect_scene_events_adaptive(
    input_path: Path,
    scene_threshold: float,
    *,
    duration_seconds: float | None = None,
    show_progress: bool = False,
    progress_label: str | None = "Analyzing scenes",
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[list[float], float]:
    thresholds = _adaptive_scene_thresholds(scene_threshold)
    detected_events: list[float] = []
    selected_threshold = thresholds[0]
    for index, candidate_threshold in enumerate(thresholds):
        attempt_label = progress_label
        if show_progress and progress_label is not None and index > 0:
            attempt_label = f"{progress_label} retry {index + 1}"
        detected_events = detect_scene_events(
            input_path,
            candidate_threshold,
            duration_seconds=duration_seconds,
            show_progress=show_progress,
            progress_label=attempt_label,
            progress_callback=progress_callback,
        )
        selected_threshold = candidate_threshold
        if detected_events:
            if index > 0:
                if show_progress and progress_label is not None:
                    print(file=sys.stderr)
                print(
                    (
                        f"Scene threshold adjusted to {candidate_threshold:.3f}; "
                        f"detected {len(detected_events)} scene events."
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            return detected_events, selected_threshold
        if index + 1 < len(thresholds):
            next_threshold = thresholds[index + 1]
            if show_progress and progress_label is not None:
                print(file=sys.stderr)
            print(
                (
                    f"No scene events at threshold {candidate_threshold:.3f}; "
                    f"retrying with {next_threshold:.3f}."
                ),
                file=sys.stderr,
                flush=True,
            )
    return detected_events, selected_threshold


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
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _parse_timestamp_seconds(value: str) -> float | None:
    match = WHISPER_TIMESTAMP_PATTERN.match(value.strip())
    if not match:
        return None
    hours_text, minutes_text, seconds_text, millis_text = match.groups()
    try:
        hours = int(hours_text)
        minutes = int(minutes_text)
        seconds = int(seconds_text)
        millis = int((millis_text or "0").ljust(3, "0")[:3])
    except ValueError:
        return None
    return max(0.0, hours * 3600 + minutes * 60 + seconds + millis / 1000.0)


def _extract_whisper_segment_bounds(raw_segment: dict[str, object]) -> tuple[float, float] | None:
    offsets = raw_segment.get("offsets")
    if isinstance(offsets, dict):
        offset_start_ms = _coerce_float(offsets.get("from"))
        offset_end_ms = _coerce_float(offsets.get("to"))
        if (
            offset_start_ms is not None
            and offset_end_ms is not None
            and offset_end_ms > offset_start_ms
        ):
            return max(0.0, offset_start_ms / 1000.0), max(0.0, offset_end_ms / 1000.0)

    timestamps = raw_segment.get("timestamps")
    if isinstance(timestamps, dict):
        start_text = timestamps.get("from")
        end_text = timestamps.get("to")
        if isinstance(start_text, str) and isinstance(end_text, str):
            start_seconds = _parse_timestamp_seconds(start_text)
            end_seconds = _parse_timestamp_seconds(end_text)
            if (
                start_seconds is not None
                and end_seconds is not None
                and end_seconds > start_seconds
            ):
                return start_seconds, end_seconds

    direct_start = _coerce_float(raw_segment.get("start"))
    direct_end = _coerce_float(raw_segment.get("end"))
    if direct_start is not None and direct_end is not None and direct_end > direct_start:
        return max(0.0, direct_start), max(0.0, direct_end)

    t0 = _coerce_float(raw_segment.get("t0"))
    t1 = _coerce_float(raw_segment.get("t1"))
    if t0 is not None and t1 is not None and t1 > t0:
        # whisper.cpp historic t0/t1 values are in 10ms increments.
        return max(0.0, t0 / 100.0), max(0.0, t1 / 100.0)

    return None


def _score_transcript_text(text: str) -> tuple[float, tuple[str, ...]]:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if not normalized:
        return 0.0, ()
    # Ignore common non-speech placeholders emitted by some models.
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
    for pattern, bonus in regex_bonuses:
        if re.search(pattern, normalized):
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

    punctuation_boost = min(0.24, text.count("!") * 0.05)
    uppercase_chars = sum(1 for char in text if char.isupper())
    alpha_chars = sum(1 for char in text if char.isalpha())
    uppercase_ratio = uppercase_chars / alpha_chars if alpha_chars > 0 else 0.0
    if uppercase_ratio >= 0.38 and alpha_chars >= 8:
        raw_score += min(0.18, uppercase_ratio * 0.35)
    raw_score += punctuation_boost

    if not semantic_signal:
        return 0.0, ()

    score = max(0.0, min(1.0, math.tanh(raw_score * 0.72)))
    return score, tuple(sorted(set(keyword_hits)))


def _adaptive_ai_cue_thresholds(
    requested_threshold: float,
    *,
    minimum_threshold: float = ADAPTIVE_AI_CUE_THRESHOLD_MIN,
) -> list[float]:
    clamped_threshold = min(1.0, max(0.0, requested_threshold))
    thresholds: list[float] = []
    for factor in ADAPTIVE_AI_CUE_THRESHOLD_FACTORS:
        candidate = max(minimum_threshold, clamped_threshold * factor)
        if any(abs(candidate - existing) < 1e-6 for existing in thresholds):
            continue
        thresholds.append(candidate)
    if minimum_threshold not in thresholds:
        thresholds.append(minimum_threshold)
    thresholds.sort(reverse=True)
    return thresholds


def _parse_whisper_json_cues(
    output_json_path: Path,
    *,
    cue_threshold: float,
) -> list[TranscriptionCue]:
    try:
        data = json.loads(output_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EditorError(
            f"Could not parse local AI transcript JSON: {output_json_path}"
        ) from error

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
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            continue
        bounds = _extract_whisper_segment_bounds(raw_segment)
        if bounds is None:
            continue
        start, end = bounds
        if end <= start:
            continue
        text_value = raw_segment.get("text", "")
        text = text_value.strip() if isinstance(text_value, str) else str(text_value).strip()
        if not text:
            continue

        score, keyword_hits = _score_transcript_text(text)
        if score < cue_threshold:
            continue
        cues.append(
            TranscriptionCue(
                start=start,
                end=end,
                score=score,
                text=text,
                keywords=keyword_hits,
            )
        )

    cues.sort(key=lambda cue: (cue.start, -cue.score))
    deduped: list[TranscriptionCue] = []
    for cue in cues:
        if deduped and cue.start - deduped[-1].start < 0.75:
            if cue.score > deduped[-1].score:
                deduped[-1] = cue
            continue
        deduped.append(cue)
    return deduped


def _resolve_whisper_cpp_binary(configured_binary: str) -> str | None:
    binary_value = configured_binary.strip()
    if binary_value and binary_value != "auto":
        resolved = shutil.which(binary_value)
        if resolved is not None:
            return resolved
        if Path(binary_value).exists():
            return binary_value
        return None

    for candidate in ("whisper-cli", "whisper-cpp", "main"):
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved
    return None


def _resolve_tesseract_binary(configured_binary: str) -> str | None:
    binary_value = configured_binary.strip()
    if binary_value and binary_value != "auto":
        resolved = shutil.which(binary_value)
        if resolved is not None:
            return resolved
        if Path(binary_value).exists():
            return binary_value
        return None
    return shutil.which("tesseract")


def _extract_audio_for_whisper(
    *,
    input_path: Path,
    output_audio_path: Path,
    audio_stream_index: int | None = None,
) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
    ]
    if audio_stream_index is not None:
        cmd.extend(["-map", f"0:a:{audio_stream_index}"])
    cmd.extend(
        [
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(output_audio_path),
        ]
    )
    _run_command(cmd, capture_output=True)


def _extract_ocr_frames(
    *,
    input_path: Path,
    output_dir: Path,
    sample_fps: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "frame_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        (
            f"fps={sample_fps:.3f},"
            "scale=1280:-1:flags=fast_bilinear,"
            "crop=iw:trunc(ih*0.42):0:0"
        ),
        "-q:v",
        "5",
        str(output_pattern),
    ]
    _run_command(cmd, capture_output=True)
    return sorted(output_dir.glob("frame_*.jpg"))


def _run_tesseract_ocr(
    *,
    image_path: Path,
    tesseract_binary: str,
) -> str:
    cmd = [
        tesseract_binary,
        str(image_path),
        "stdout",
        "--psm",
        "6",
        "-l",
        "eng",
    ]
    result = _run_command(cmd, capture_output=True)
    return (result.stdout or "").strip()


def _collect_ocr_cues(
    *,
    input_path: Path,
    tesseract_binary: str,
    sample_fps: float,
    cue_threshold: float,
) -> list[TranscriptionCue]:
    resolved_binary = _resolve_tesseract_binary(tesseract_binary)
    if resolved_binary is None:
        raise EditorError(
            "Could not find tesseract binary. Install tesseract or pass --tesseract-bin."
        )
    if sample_fps <= 0:
        return []

    with tempfile.TemporaryDirectory(prefix="lol-ocr-") as temp_dir:
        frames_dir = Path(temp_dir) / "frames"
        frame_paths = _extract_ocr_frames(
            input_path=input_path,
            output_dir=frames_dir,
            sample_fps=sample_fps,
        )
        if not frame_paths:
            return []

        cues: list[TranscriptionCue] = []
        for index, frame_path in enumerate(frame_paths):
            text = _run_tesseract_ocr(
                image_path=frame_path,
                tesseract_binary=resolved_binary,
            )
            if not text:
                continue
            score, keyword_hits = _score_transcript_text(text)
            if score < cue_threshold:
                continue
            center = index / sample_fps
            start = max(0.0, center - 0.8)
            end = center + 2.2
            cues.append(
                TranscriptionCue(
                    start=start,
                    end=end,
                    score=score,
                    text=text,
                    keywords=keyword_hits,
                )
            )

    cues.sort(key=lambda cue: (cue.start, -cue.score))
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

    resolved_binary = _resolve_tesseract_binary(tesseract_binary)
    if resolved_binary is None:
        return None

    start_time = max(0.0, duration_seconds - max(60.0, tail_seconds))
    with tempfile.TemporaryDirectory(prefix="lol-result-ocr-") as temp_dir:
        frames_dir = Path(temp_dir) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        output_pattern = frames_dir / "frame_%06d.jpg"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-ss",
            f"{start_time:.3f}",
            "-i",
            str(input_path),
            "-vf",
            (
                f"fps={sample_fps:.3f},"
                "scale=1280:-1:flags=fast_bilinear,"
                "crop=iw:trunc(ih*0.62):0:0"
            ),
            "-q:v",
            "5",
            str(output_pattern),
        ]
        _run_command(cmd, capture_output=True)
        frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
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
            text = _run_tesseract_ocr(
                image_path=frame_path,
                tesseract_binary=resolved_binary,
            )
            if not text:
                continue
            normalized = re.sub(r"\s+", " ", text.lower())
            frame_time = start_time + (index / sample_fps)
            if result_pattern.search(normalized):
                result_hits.append(frame_time)
            if nexus_pattern.search(normalized) and terminal_nexus_pattern.search(normalized):
                nexus_hits.append(frame_time)

    if result_hits:
        result_time = result_hits[-1]
        return min(duration_seconds, result_time + 2.5)
    if nexus_hits:
        return min(duration_seconds, nexus_hits[-1] + 4.0)
    return None


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
    if not whisper_model.exists() or not whisper_model.is_file():
        raise EditorError(f"Whisper model file not found: {whisper_model}")

    resolved_binary = _resolve_whisper_cpp_binary(whisper_binary)
    if resolved_binary is None:
        raise EditorError(
            "Could not find whisper.cpp binary. Install whisper.cpp and ensure "
            "`whisper-cli` is on PATH, or pass --whisper-bin with a full path."
        )

    with tempfile.TemporaryDirectory(prefix="lol-whisper-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        audio_path = temp_dir_path / "transcription-audio.wav"
        try:
            _extract_audio_for_whisper(
                input_path=input_path,
                output_audio_path=audio_path,
                audio_stream_index=whisper_audio_stream,
            )
        except subprocess.CalledProcessError as error:
            stderr_text = (error.stderr or "").strip()
            detail = f"\n{stderr_text}" if stderr_text else ""
            raise EditorError(
                "Could not extract audio for local AI transcription."
                f"{detail}"
            ) from error

        output_prefix = temp_dir_path / "transcript"
        cmd = [
            resolved_binary,
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
            if not whisper_vad_model.exists() or not whisper_vad_model.is_file():
                raise EditorError(f"Whisper VAD model file not found: {whisper_vad_model}")
            cmd.append("--vad")
            cmd.extend(["--vad-model", str(whisper_vad_model)])
            cmd.extend(["--vad-threshold", f"{whisper_vad_threshold:.3f}"])
        result = _run_command(cmd, capture_output=True)
        output_json_path = Path(f"{output_prefix}.json")
        if not output_json_path.exists():
            json_candidates = sorted(temp_dir_path.glob("*.json"))
            if json_candidates:
                output_json_path = json_candidates[0]
            else:
                whisper_output = ((result.stderr or "") + "\n" + (result.stdout or "")).strip()
                output_tail = whisper_output[-1200:] if whisper_output else ""
                detail = f"\nwhisper-cli output:\n{output_tail}" if output_tail else ""
                raise EditorError(
                    "whisper.cpp completed but did not produce a JSON transcript. "
                    "Ensure your whisper-cli supports --output-json."
                    f"{detail}"
                )
        return _parse_whisper_json_cues(
            output_json_path,
            cue_threshold=cue_threshold,
        )


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
        best_contribution = 0.0
        for cue in ai_cues:
            cue_center = cue.center
            if cue_center < window.start - clamped_radius or cue_center > window.end + clamped_radius:
                continue
            overlap_seconds = max(0.0, min(window.end, cue.end) - max(window.start, cue.start))
            overlap_ratio = overlap_seconds / window_duration
            distance = abs(window_center - cue_center)
            proximity = max(0.0, 1.0 - distance / clamped_radius)
            contribution = cue.score * (0.72 * proximity + 0.28 * overlap_ratio)
            if contribution > best_contribution:
                best_contribution = contribution

        boosted_score = min(1.0, window.score + 0.38 * best_contribution)
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
    if not ai_cues:
        return []
    return sorted(
        {
            cue.center
            for cue in ai_cues
            if math.isfinite(cue.center) and cue.center >= 0.0
        }
    )


def _detect_watchability_crop_filter(
    input_path: Path,
    *,
    duration_seconds: float,
) -> str | None:
    sample_duration = max(0.0, min(duration_seconds, 300.0))
    if sample_duration <= 0:
        return None
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-ss",
        "0",
        "-t",
        f"{sample_duration:.3f}",
        "-i",
        str(input_path),
        "-vf",
        "fps=0.500,cropdetect=limit=0.08:round=2:reset=0",
        "-an",
        "-f",
        "null",
        "-",
    ]
    try:
        result = _run_command(cmd, capture_output=True)
    except subprocess.CalledProcessError:
        return None

    output = (result.stderr or "") + "\n" + (result.stdout or "")
    matches = CROPDETECT_PATTERN.findall(output)
    if not matches:
        return None

    crop_counts: dict[tuple[int, int, int, int], int] = {}
    for width_text, height_text, x_text, y_text in matches:
        width = int(width_text)
        height = int(height_text)
        x_offset = int(x_text)
        y_offset = int(y_text)
        if width <= 0 or height <= 0 or x_offset < 0 or y_offset < 0:
            continue
        crop_counts[(width, height, x_offset, y_offset)] = (
            crop_counts.get((width, height, x_offset, y_offset), 0) + 1
        )

    if not crop_counts:
        return None

    best_crop, _ = max(
        crop_counts.items(),
        key=lambda item: (item[1], item[0][0] * item[0][1]),
    )
    width, height, x_offset, y_offset = best_crop
    return f"crop={width}:{height}:{x_offset}:{y_offset}"


def _extract_signalstats_samples(
    input_path: Path,
    *,
    duration_seconds: float,
    sample_fps: float,
    show_progress: bool,
    progress_label: str | None = "Scoring gameplay",
    progress_callback: Callable[[float], None] | None = None,
) -> list[tuple[float, float, float]]:
    if show_progress and progress_callback is not None:
        progress_callback(0.01)
    crop_filter = _detect_watchability_crop_filter(
        input_path,
        duration_seconds=duration_seconds,
    )
    if show_progress and progress_callback is not None:
        progress_callback(0.06)

    signal_filters = []
    if crop_filter is not None:
        signal_filters.append(crop_filter)
    signal_filters.extend(
        [
            f"fps={sample_fps:.3f}",
            "scale=320:-1:flags=fast_bilinear",
            "signalstats",
            "metadata=print",
        ]
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        str(input_path),
        "-vf",
        ",".join(signal_filters),
        "-an",
        "-f",
        "null",
        "-",
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    samples: list[tuple[float, float, float]] = []
    stderr_tail: deque[str] = deque(maxlen=120)
    current_time: float | None = None
    current_motion: float | None = None
    current_saturation: float | None = None
    last_output_ratio = -1.0
    try:
        assert process.stderr is not None
        for line in process.stderr:
            stderr_tail.append(line)

            frame_match = SIGNALSTATS_FRAME_PATTERN.search(line)
            if frame_match:
                if (
                    current_time is not None
                    and current_motion is not None
                    and current_saturation is not None
                ):
                    samples.append((current_time, current_motion, current_saturation))
                try:
                    current_time = float(frame_match.group(1))
                except ValueError:
                    current_time = None
                current_motion = None
                current_saturation = None

            motion_match = SIGNALSTATS_YDIF_PATTERN.search(line)
            if motion_match:
                try:
                    current_motion = float(motion_match.group(1))
                except ValueError:
                    current_motion = None

            saturation_match = SIGNALSTATS_SATAVG_PATTERN.search(line)
            if saturation_match:
                try:
                    current_saturation = float(saturation_match.group(1))
                except ValueError:
                    current_saturation = None

            if show_progress and duration_seconds > 0:
                progress_time = current_time
                time_match = FFMPEG_TIME_PATTERN.search(line)
                if progress_time is None and time_match:
                    progress_time = _parse_ffmpeg_clock_to_seconds(time_match.group(1))
                if progress_time is not None:
                    progress_ratio = min(1.0, max(0.0, progress_time / duration_seconds))
                    if progress_ratio >= last_output_ratio + 0.01 or progress_ratio >= 1.0:
                        if progress_label is not None:
                            print(
                                _render_progress_line(progress_label, progress_ratio),
                                end="",
                                file=sys.stderr,
                                flush=True,
                            )
                        if progress_callback is not None:
                            progress_callback(progress_ratio)
                        last_output_ratio = progress_ratio

        if current_time is not None and current_motion is not None and current_saturation is not None:
            samples.append((current_time, current_motion, current_saturation))

        return_code = process.wait()
        if show_progress and duration_seconds > 0:
            if progress_label is not None:
                print(
                    _render_progress_line(progress_label, 1.0),
                    file=sys.stderr,
                    flush=True,
                )
            if progress_callback is not None:
                progress_callback(1.0)
        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code,
                cmd,
                stderr="".join(stderr_tail),
            )
        return samples
    except KeyboardInterrupt:
        process.terminate()
        raise


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

    sorted_events = sorted(value for value in events if value >= 0)
    windows_raw: list[tuple[float, float, float, float, float]] = []
    max_start = max(0.0, duration_seconds - window_seconds)
    start = 0.0
    while start <= max_start + 1e-6:
        end = min(duration_seconds, start + window_seconds)
        samples_in_window = [
            sample
            for sample in frame_samples
            if start <= sample[0] < end and math.isfinite(sample[1]) and math.isfinite(sample[2])
        ]
        if samples_in_window:
            motion = sum(sample[1] for sample in samples_in_window) / len(samples_in_window)
            saturation = sum(sample[2] for sample in samples_in_window) / len(samples_in_window)
            scene_hits = sum(1 for event in sorted_events if start <= event < end)
            scene_density = scene_hits / max(1.0, window_seconds)
            windows_raw.append((start, end, motion, saturation, scene_density))
        start += step_seconds

    if not windows_raw:
        return []

    motions = [window[2] for window in windows_raw]
    saturations = [window[3] for window in windows_raw]
    densities = [window[4] for window in windows_raw]
    motion_low = _percentile(motions, 0.10)
    motion_high = _percentile(motions, 0.90)
    density_high = max(densities) if any(density > 0 for density in densities) else 1.0
    sat_low = _percentile(saturations, 0.15)
    sat_high = _percentile(saturations, 0.90)
    low_motion_threshold = _percentile(motions, 0.25)
    low_density_threshold = _percentile(densities, 0.25)
    low_saturation_threshold = _percentile(saturations, 0.20)

    scored_windows: list[VisionWindow] = []
    for start, end, motion, saturation, scene_density in windows_raw:
        motion_norm = _normalize(motion, low=motion_low, high=motion_high)
        scene_norm = _normalize(scene_density, low=0.0, high=density_high)
        saturation_norm = _normalize(saturation, low=sat_low, high=sat_high)
        late_game_weight = 0.9 + 0.2 * (end / duration_seconds)

        score = (0.50 * motion_norm + 0.35 * scene_norm + 0.15 * saturation_norm) * late_game_weight
        if motion <= low_motion_threshold and scene_density <= low_density_threshold:
            score -= 0.10
        if saturation <= low_saturation_threshold:
            score -= 0.12
        score = min(1.0, max(0.0, score))

        scored_windows.append(
            VisionWindow(
                start=start,
                end=end,
                score=score,
                motion=motion,
                saturation=saturation,
                scene_density=scene_density,
            )
        )

    return scored_windows


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
) -> list[VisionWindow]:
    samples = _extract_signalstats_samples(
        input_path,
        duration_seconds=duration_seconds,
        sample_fps=sample_fps,
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


def _rank_vision_candidates(
    vision_windows: list[VisionWindow],
    *,
    min_gap_seconds: float,
    clip_before: float,
    clip_after: float,
) -> list[float]:
    if not vision_windows:
        return []

    ranked = sorted(vision_windows, key=lambda window: (-window.score, window.start))
    selected: list[float] = []
    # Respect user pacing controls while still avoiding near-duplicate windows.
    spacing = max(min_gap_seconds, (clip_before + clip_after) * 0.25)
    for window in ranked:
        center = (window.start + window.end) / 2
        if any(abs(center - value) < spacing for value in selected):
            continue
        selected.append(center)
    return selected


def _detect_gameplay_start(
    vision_windows: list[VisionWindow],
    *,
    duration_seconds: float,
) -> float:
    if not vision_windows or duration_seconds <= 0:
        return 0.0

    ordered = sorted(vision_windows, key=lambda window: window.start)
    search_limit = min(duration_seconds * 0.6, 900.0)
    candidates = [window for window in ordered if window.start <= search_limit]
    if len(candidates) < 2:
        candidates = ordered

    scores = [window.score for window in candidates]
    motions = [window.motion for window in candidates]
    saturations = [window.saturation for window in candidates]
    densities = [window.scene_density for window in candidates]

    early_window_count = min(len(candidates), 12)
    early_motion = motions[:early_window_count]
    early_scores = scores[:early_window_count]
    baseline_motion = _percentile(early_motion, 0.50) if early_motion else _percentile(motions, 0.25)
    baseline_score = _percentile(early_scores, 0.50) if early_scores else _percentile(scores, 0.25)

    score_jump_threshold = baseline_score + 0.04 if baseline_score <= 0.20 else 0.0
    motion_jump_threshold = baseline_motion * 1.8 + 0.15 if baseline_motion <= 2.0 else 0.0

    score_threshold = max(0.10, _percentile(scores, 0.30), score_jump_threshold)
    motion_threshold = max(0.45, _percentile(motions, 0.25), motion_jump_threshold)
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
            next_window = candidates[index + 1]
            sustained = (
                next_window.score >= score_threshold * 0.7
                or next_window.motion >= motion_threshold * 0.8
            )
            if not sustained:
                continue

        return max(0.0, window.start - 3.0)

    return 0.0


def _detect_death_cues(
    vision_windows: list[VisionWindow],
    *,
    min_spacing_seconds: float = 35.0,
) -> list[float]:
    if len(vision_windows) < 2:
        return []

    ordered = sorted(vision_windows, key=lambda window: window.start)
    saturations = [window.saturation for window in ordered]
    motions = [window.motion for window in ordered]
    densities = [window.scene_density for window in ordered]
    sat_p30 = _percentile(saturations, 0.30)
    gray_threshold = max(7.0, min(18.0, sat_p30 + 2.0))
    motion_soft_cap = _percentile(motions, 0.65)
    density_soft_cap = _percentile(densities, 0.60)
    score_soft_cap = _percentile([window.score for window in ordered], 0.55)

    def looks_dead(window: VisionWindow) -> bool:
        return (
            window.saturation <= gray_threshold
            and window.score <= score_soft_cap
            and (window.motion <= motion_soft_cap or window.scene_density <= density_soft_cap)
        )

    death_cues: list[float] = []
    for index in range(1, len(ordered)):
        previous = ordered[index - 1]
        current = ordered[index]
        if looks_dead(previous) or not looks_dead(current):
            continue

        has_persistence = index + 1 < len(ordered) and looks_dead(ordered[index + 1])
        if not has_persistence:
            continue

        lead_in_seconds = max(10.0, (current.end - current.start) * 0.75)
        cue_time = max(0.0, current.start - lead_in_seconds)
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

    ordered = sorted(vision_windows, key=lambda window: window.start)
    scores = [window.score for window in ordered]
    motions = [window.motion for window in ordered]
    densities = [window.scene_density for window in ordered]
    saturations = [window.saturation for window in ordered]

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
        next_window = ordered[index + 1] if index + 1 < len(ordered) else None
        has_rise = (
            previous is None
            or previous.score < current.score * 0.92
            or previous.motion < motion_threshold * 0.9
        )
        has_follow_through = (
            next_window is None
            or next_window.score >= score_threshold * 0.72
            or next_window.motion >= motion_threshold * 0.78
        )
        if not has_rise or not has_follow_through:
            continue

        lead_in_seconds = max(12.0, (current.end - current.start) * 0.9)
        cue_time = max(0.0, current.start - lead_in_seconds)
        if combat_cues and cue_time - combat_cues[-1] < min_spacing_seconds:
            continue
        combat_cues.append(cue_time)

    return combat_cues


def sample_evenly(values: list[float], max_items: int) -> list[float]:
    if max_items <= 0:
        return []
    if len(values) <= max_items:
        return values[:]
    if max_items == 1:
        return [values[len(values) // 2]]

    sampled: list[float] = []
    for index in range(max_items):
        position = round(index * (len(values) - 1) / (max_items - 1))
        sampled.append(values[position])
    deduped = list(dict.fromkeys(sampled))
    if len(deduped) == max_items:
        return deduped
    for value in values:
        if len(deduped) == max_items:
            break
        if value not in deduped:
            deduped.append(value)
    return sorted(deduped)


def merge_segments(segments: list[Segment], merge_gap: float = 0.25) -> list[Segment]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda segment: segment.start)
    merged = [ordered[0]]
    for segment in ordered[1:]:
        last = merged[-1]
        if segment.start <= last.end + merge_gap:
            merged[-1] = Segment(last.start, max(last.end, segment.end))
        else:
            merged.append(segment)
    return merged


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
        late_game_weight = 1.0 + (event / duration_seconds if duration_seconds > 0 else 0.0)
        scored.append((local_density * late_game_weight, event))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [event for _, event in scored]


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
    for index in range(1, grid_count + 1):
        center = start + step * index
        if any(abs(center - value) < min_gap_seconds for value in existing + selected):
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
    for index in range(remaining):
        selected.append(start + spacing * (index + 1))
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
            # Treat the last near-end scene event as the terminal in-game moment.
            # This avoids trailing into post-game screens after nexus destruction.
            likely_finish = min(duration_seconds, max(0.0, last_event))

    outro_end = max(0.0, likely_finish)
    outro_start = max(0.0, outro_end - outro_seconds)
    if outro_end <= outro_start:
        return None
    return Segment(start=outro_start, end=outro_end)


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
) -> tuple[list[Segment], bool]:
    if duration_seconds <= 0 or max_clips <= 0:
        return [], False

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

    anchored_segments = [segment for segment in (intro_segment, outro_segment) if segment is not None]
    if (
        intro_segment is not None
        and outro_segment is not None
        and intro_segment.end >= outro_segment.start
    ):
        return [Segment(intro_segment.start, duration_seconds)], False

    interior_start = intro_segment.end if intro_segment is not None else 0.0
    interior_end = outro_segment.start if outro_segment is not None else duration_seconds
    clip_window_seconds = clip_before + clip_after
    target_total_duration = (
        duration_seconds
        if target_duration_seconds <= 0
        else min(duration_seconds, target_duration_seconds)
    )
    anchored_duration = sum(segment.duration for segment in anchored_segments)
    target_middle_duration = max(0.0, target_total_duration - anchored_duration)

    middle_clip_target = 0
    if clip_window_seconds > 0 and interior_end > interior_start and target_middle_duration > 0:
        middle_clip_target = min(max_clips, math.ceil(target_middle_duration / clip_window_seconds))

    middle_segments: list[Segment] = []
    used_fallback = False
    spacing_seconds = min_gap_seconds
    death_cues = _detect_death_cues(vision_windows or [])
    combat_cues = _detect_combat_cues(vision_windows or [])
    ai_cue_details: list[tuple[float, float]] = []
    if ai_priority_details:
        for cue in ai_priority_details:
            center = cue.center
            if math.isfinite(center) and center >= 0.0:
                ai_cue_details.append((center, max(0.0, min(1.0, cue.score))))
    elif ai_priority_cues:
        ai_cue_details.extend(
            (cue_time, 0.40)
            for cue_time in ai_priority_cues
            if math.isfinite(cue_time) and cue_time >= 0.0
        )
    ai_cue_details.sort(key=lambda cue: cue[0])
    forced_cues = sorted(
        [("death", cue_time, 0.92) for cue_time in death_cues]
        + [("combat", cue_time, 0.86) for cue_time in combat_cues]
        + [("ai", cue_time, 0.55 + 0.45 * cue_score) for cue_time, cue_score in ai_cue_details],
        key=lambda cue: cue[1],
    )
    if middle_clip_target > 0:
        effective_clip_before = clip_before
        effective_clip_after = clip_after
        if clip_window_seconds > 0 and target_middle_duration > 0:
            target_per_clip = target_middle_duration / middle_clip_target
            if target_per_clip > clip_window_seconds:
                expansion_factor = min(1.35, target_per_clip / clip_window_seconds)
                effective_clip_before *= expansion_factor
                effective_clip_after *= expansion_factor

        if vision_windows:
            ranked_events = _rank_vision_candidates(
                vision_windows,
                min_gap_seconds=min_gap_seconds,
                clip_before=effective_clip_before,
                clip_after=effective_clip_after,
            )
        else:
            ranked_events = _rank_event_candidates(
                events,
                min_gap_seconds=min_gap_seconds,
                duration_seconds=duration_seconds,
                    clip_before=effective_clip_before,
                    clip_after=effective_clip_after,
                )

        forced_centers: list[float] = []
        forced_segments: list[Segment] = []
        candidate_forced_cues = forced_cues
        normalized_forced_share = min(0.95, max(0.20, forced_cue_share))
        max_forced_slots = max(1, math.ceil(middle_clip_target * normalized_forced_share))
        if len(candidate_forced_cues) > max_forced_slots:
            timeline_indexes = sample_evenly(
                list(range(len(candidate_forced_cues))),
                max_forced_slots,
            )
            selected_forced = [candidate_forced_cues[index] for index in timeline_indexes]
            priority_ranked = sorted(candidate_forced_cues, key=lambda cue: (-cue[2], cue[1]))
            for cue in priority_ranked:
                if cue in selected_forced:
                    continue
                weakest_index = min(range(len(selected_forced)), key=lambda idx: selected_forced[idx][2])
                if cue[2] > selected_forced[weakest_index][2] + 0.05:
                    selected_forced[weakest_index] = cue
            candidate_forced_cues = sorted(selected_forced, key=lambda cue: cue[1])
        if len(candidate_forced_cues) > middle_clip_target:
            sampled_indexes = sample_evenly(
                list(range(len(candidate_forced_cues))),
                middle_clip_target,
            )
            candidate_forced_cues = [candidate_forced_cues[index] for index in sampled_indexes]

        forced_spacing_seconds = max(6.0, spacing_seconds * 0.72)
        for cue_type, cue_time, cue_priority in candidate_forced_cues:
            if cue_time <= interior_start or cue_time >= interior_end:
                continue
            if any(abs(cue_time - chosen) < forced_spacing_seconds for chosen in forced_centers):
                continue
            if cue_type == "death":
                cue_before = max(effective_clip_before, EVENT_CONTEXT_DEATH_BEFORE_SECONDS)
                cue_after = max(effective_clip_after, EVENT_CONTEXT_DEATH_AFTER_SECONDS)
            elif cue_type == "combat":
                cue_before = max(effective_clip_before, EVENT_CONTEXT_COMBAT_BEFORE_SECONDS)
                cue_after = max(effective_clip_after, EVENT_CONTEXT_COMBAT_AFTER_SECONDS)
            else:
                cue_before = max(effective_clip_before, EVENT_CONTEXT_AI_BEFORE_SECONDS)
                cue_after = max(effective_clip_after, EVENT_CONTEXT_AI_AFTER_SECONDS)
                if cue_priority >= 0.86:
                    cue_before = max(cue_before, effective_clip_before + 3.0)
                    cue_after = max(cue_after, effective_clip_after + 4.0)
                elif cue_priority >= 0.72:
                    cue_before = max(cue_before, effective_clip_before + 1.5)
                    cue_after = max(cue_after, effective_clip_after + 2.5)
            segment = _build_segment_from_center(
                cue_time,
                clip_before=cue_before,
                clip_after=cue_after,
                clamp_start=interior_start,
                clamp_end=interior_end,
            )
            if segment is None:
                continue
            forced_centers.append(cue_time)
            forced_segments.append(segment)
            if len(forced_centers) >= middle_clip_target:
                break

        middle_segments.extend(forced_segments)
        remaining_slots = max(0, middle_clip_target - len(forced_centers))
        selected_centers: list[float] = []
        occupied_centers = forced_centers[:]

        for event in ranked_events:
            if len(selected_centers) >= remaining_slots:
                break
            if event <= interior_start or event >= interior_end:
                continue
            if any(abs(event - chosen) < spacing_seconds for chosen in occupied_centers):
                continue
            selected_centers.append(event)
            occupied_centers.append(event)

        if len(selected_centers) < remaining_slots:
            used_fallback = True
            missing = remaining_slots - len(selected_centers)
            selected_centers.extend(
                _generate_fallback_centers(
                    start=interior_start,
                    end=interior_end,
                    count=missing,
                    min_gap_seconds=spacing_seconds,
                    existing=occupied_centers + selected_centers,
                )
            )

        for center in sorted(selected_centers):
            segment = _build_segment_from_center(
                center,
                clip_before=effective_clip_before,
                clip_after=effective_clip_after,
                clamp_start=interior_start,
                clamp_end=interior_end,
            )
            if segment is not None:
                middle_segments.append(segment)

    all_segments = anchored_segments + middle_segments
    merge_gap = max(0.25, min_gap_seconds * (0.35 if vision_windows else 0.15))
    return merge_segments(all_segments, merge_gap=merge_gap), used_fallback


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
) -> None:
    payload = {
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
    if ai_cues:
        payload["ai_cues"] = [
            {
                "start": round(cue.start, 3),
                "end": round(cue.end, 3),
                "center": round(cue.center, 3),
                "score": round(cue.score, 4),
                "keywords": list(cue.keywords),
                "text": cue.text[:200],
            }
            for cue in ai_cues
        ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        rendered_json = json.dumps(payload, indent=2, allow_nan=False)
    except ValueError as error:
        raise EditorError("Plan contains non-finite numeric values.") from error
    output_path.write_text(rendered_json, encoding="utf-8")


def read_plan(plan_path: Path) -> list[Segment]:
    try:
        raw_text = plan_path.read_text(encoding="utf-8")
    except OSError as error:
        raise EditorError(f"Could not read plan file: {plan_path}") from error

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise EditorError(f"Plan file is not valid JSON: {plan_path}") from error

    if not isinstance(data, dict):
        raise EditorError(f"Plan JSON must be an object: {plan_path}")

    raw_segments = data.get("segments", [])
    if not isinstance(raw_segments, list):
        raise EditorError(f"'segments' must be a list in plan: {plan_path}")

    segments: list[Segment] = []
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            raise EditorError(
                f"Segment at index {index} is not an object in plan: {plan_path}"
            )
        try:
            start = float(raw["start"])
            end = float(raw["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise EditorError(
                f"Segment at index {index} must include numeric 'start' and 'end' in plan: {plan_path}"
            ) from error
        if not math.isfinite(start) or not math.isfinite(end):
            raise EditorError(
                f"Segment at index {index} has non-finite 'start'/'end' in plan: {plan_path}"
            )
        if start < 0 or end < 0:
            raise EditorError(
                f"Segment at index {index} has negative 'start'/'end' in plan: {plan_path}"
            )
        if end <= start:
            raise EditorError(
                f"Segment at index {index} must have end > start in plan: {plan_path}"
            )
        segments.append(Segment(start=start, end=end))
    return segments


def _build_filter_complex(
    segments: list[Segment],
    include_audio: bool,
    *,
    include_video: bool = True,
    allow_upscale: bool = False,
    crop_filter: str | None = None,
    crossfade_seconds: float = 0.0,
    audio_fade_seconds: float = 0.03,
    audio_post_filter: str | None = None,
) -> tuple[str, str | None, str | None]:
    pieces: list[str] = []
    durations = [segment.duration for segment in segments]

    video_output_label: str | None = None
    if include_video:
        for index, segment in enumerate(segments):
            pieces.append(
                f"[0:v]trim=start={segment.start:.3f}:end={segment.end:.3f},setpts=PTS-STARTPTS[v{index}]"
            )

        if crossfade_seconds > 0 and len(segments) > 1:
            running_duration = durations[0]
            current_label = "v0"
            for index in range(1, len(segments)):
                transition_duration = _effective_transition_duration(
                    crossfade_seconds,
                    left_duration=running_duration,
                    right_duration=durations[index],
                )
                offset = max(0.0, running_duration - transition_duration)
                next_label = f"vxf{index}"
                pieces.append(
                    f"[{current_label}][v{index}]"
                    f"xfade=transition=fade:duration={transition_duration:.3f}:offset={offset:.3f}"
                    f"[{next_label}]"
                )
                current_label = next_label
                running_duration = running_duration + durations[index] - transition_duration
            video_source = current_label
        else:
            concat_inputs = "".join(f"[v{index}]" for index in range(len(segments)))
            pieces.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=0[vcat]")
            video_source = "vcat"

        pieces.append(
            f"[{video_source}]"
            f"{_video_postprocess_filter(allow_upscale=allow_upscale, crop_filter=crop_filter)}"
            "[vout]"
        )
        video_output_label = "[vout]"

    audio_output_label: str | None = None
    if include_audio:
        for index, segment in enumerate(segments):
            audio_steps = [
                f"atrim=start={segment.start:.3f}:end={segment.end:.3f}",
                "asetpts=PTS-STARTPTS",
            ]
            if audio_fade_seconds > 0:
                fade_duration = min(audio_fade_seconds, segment.duration / 2)
                if fade_duration > 0:
                    fade_out_start = max(0.0, segment.duration - fade_duration)
                    audio_steps.extend(
                        [
                            f"afade=t=in:st=0:d={fade_duration:.3f}",
                            f"afade=t=out:st={fade_out_start:.3f}:d={fade_duration:.3f}",
                        ]
                    )
            pieces.append(f"[0:a]{','.join(audio_steps)}[a{index}]")

        if crossfade_seconds > 0 and len(segments) > 1:
            running_duration = durations[0]
            current_label = "a0"
            for index in range(1, len(segments)):
                transition_duration = _effective_transition_duration(
                    crossfade_seconds,
                    left_duration=running_duration,
                    right_duration=durations[index],
                )
                next_label = f"axf{index}"
                pieces.append(
                    f"[{current_label}][a{index}]"
                    f"acrossfade=d={transition_duration:.3f}:c1=tri:c2=tri"
                    f"[{next_label}]"
                )
                current_label = next_label
                running_duration = running_duration + durations[index] - transition_duration
            audio_source = current_label
        else:
            concat_inputs = "".join(f"[a{index}]" for index in range(len(segments)))
            pieces.append(f"{concat_inputs}concat=n={len(segments)}:v=0:a=1[acat]")
            audio_source = "acat"

        post_filter = audio_post_filter or "anull"
        pieces.append(f"[{audio_source}]{post_filter}[aout]")
        audio_output_label = "[aout]"

    return ";".join(pieces), video_output_label, audio_output_label


def _estimate_render_duration(segments: list[Segment], *, crossfade_seconds: float) -> float:
    if not segments:
        return 0.0
    durations = [segment.duration for segment in segments]
    if crossfade_seconds <= 0 or len(durations) == 1:
        return max(0.0, sum(durations))

    running_duration = durations[0]
    for index in range(1, len(durations)):
        transition_duration = _effective_transition_duration(
            crossfade_seconds,
            left_duration=running_duration,
            right_duration=durations[index],
        )
        running_duration = running_duration + durations[index] - transition_duration
    return max(0.0, running_duration)


def _looks_like_loudnorm_nonfinite_error(error: subprocess.CalledProcessError) -> bool:
    stderr = error.stderr or ""
    stdout = error.stdout or ""
    combined_output = f"{stderr}\n{stdout}"
    return bool(LOUDNORM_NONFINITE_PATTERN.search(combined_output))


def _extract_loudnorm_json(analysis_output: str) -> dict[str, float]:
    required = {
        "input_i": "measured_I",
        "input_tp": "measured_TP",
        "input_lra": "measured_LRA",
        "input_thresh": "measured_thresh",
        "target_offset": "offset",
    }
    json_candidates: list[str] = []
    depth = 0
    object_start: int | None = None
    for index, character in enumerate(analysis_output):
        if character == "{":
            if depth == 0:
                object_start = index
            depth += 1
        elif character == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and object_start is not None:
                json_candidates.append(analysis_output[object_start : index + 1])
                object_start = None

    if not json_candidates:
        raise EditorError("Could not parse loudnorm analysis output.")

    payload: dict[str, object] | None = None
    for candidate in reversed(json_candidates):
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict) and all(key in decoded for key in required):
            payload = decoded
            break

    if payload is None:
        raise EditorError("Could not decode loudnorm analysis JSON.")

    extracted: dict[str, float] = {}
    for source_key, output_key in required.items():
        try:
            value = float(payload[source_key])
        except (TypeError, ValueError) as error:
            raise EditorError(f"Invalid '{source_key}' in loudnorm analysis output.") from error
        if not math.isfinite(value):
            raise EditorError(f"Non-finite '{source_key}' in loudnorm analysis output.")
        extracted[output_key] = value
    return extracted


def _build_two_pass_loudnorm_filter(analysis_command: list[str]) -> str:
    result = _run_command(analysis_command, capture_output=True)
    analysis_output = (result.stderr or "") + "\n" + (result.stdout or "")
    metrics = _extract_loudnorm_json(analysis_output)
    return (
        "loudnorm=I=-14:LRA=11:TP=-1.5:"
        f"measured_I={metrics['measured_I']:.3f}:"
        f"measured_TP={metrics['measured_TP']:.3f}:"
        f"measured_LRA={metrics['measured_LRA']:.3f}:"
        f"measured_thresh={metrics['measured_thresh']:.3f}:"
        f"offset={metrics['offset']:.3f}:"
        "linear=true:print_format=summary"
    )


def _select_loudnorm_filter(
    *,
    two_pass_loudnorm: bool,
    analysis_command: list[str] | None,
) -> str:
    if not two_pass_loudnorm or analysis_command is None:
        return LOUDNORM_FILTER
    try:
        return _build_two_pass_loudnorm_filter(analysis_command)
    except (EditorError, subprocess.CalledProcessError) as error:
        print(
            f"Warning: two-pass loudnorm analysis failed ({error}); using one-pass loudnorm.",
            file=sys.stderr,
        )
        return LOUDNORM_FILTER


def _run_with_loudnorm_fallback(
    command_with_loudnorm: list[str],
    command_without_loudnorm: list[str] | None,
    *,
    progress_label: str | None = None,
    progress_duration_seconds: float | None = None,
) -> None:
    if command_without_loudnorm is None:
        _run_command_with_stderr_tail(
            command_with_loudnorm,
            progress_label=progress_label,
            duration_seconds=progress_duration_seconds,
        )
        return

    try:
        _run_command_with_stderr_tail(
            command_with_loudnorm,
            progress_label=progress_label,
            duration_seconds=progress_duration_seconds,
        )
    except subprocess.CalledProcessError as error:
        if not _looks_like_loudnorm_nonfinite_error(error):
            raise
        print(
            "Warning: loudnorm failed due to non-finite audio values; retrying without loudness normalization.",
            file=sys.stderr,
        )
        _run_command_with_stderr_tail(
            command_without_loudnorm,
            progress_label=progress_label,
            duration_seconds=progress_duration_seconds,
        )


def _run_command_with_stderr_tail(
    cmd: list[str],
    *,
    tail_bytes: int = 256 * 1024,
    progress_label: str | None = None,
    duration_seconds: float | None = None,
) -> None:
    if progress_label is not None and duration_seconds is not None and duration_seconds > 0:
        stderr_tail: deque[str] = deque(maxlen=120)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        last_output_ratio = -1.0
        try:
            assert process.stderr is not None
            for line in process.stderr:
                stderr_tail.append(line)
                time_match = FFMPEG_TIME_PATTERN.search(line)
                if not time_match:
                    continue
                current_seconds = _parse_ffmpeg_clock_to_seconds(time_match.group(1))
                progress_ratio = min(1.0, max(0.0, current_seconds / duration_seconds))
                if progress_ratio >= last_output_ratio + 0.01 or progress_ratio >= 1.0:
                    print(
                        _render_progress_line(progress_label, progress_ratio),
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_output_ratio = progress_ratio

            return_code = process.wait()
            print(
                _render_progress_line(progress_label, 1.0),
                file=sys.stderr,
                flush=True,
            )
            if return_code != 0:
                raise subprocess.CalledProcessError(
                    return_code,
                    cmd,
                    stderr="".join(stderr_tail),
                )
            return
        except KeyboardInterrupt:
            process.terminate()
            raise

    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as stderr_file:
        completed = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
        )
        if completed.returncode == 0:
            return

        stderr_file.flush()
        file_size = stderr_file.tell()
        start = max(0, file_size - tail_bytes)
        stderr_file.seek(start, os.SEEK_SET)
        stderr_tail = stderr_file.read()
        raise subprocess.CalledProcessError(
            completed.returncode,
            cmd,
            stderr=stderr_tail,
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
    segments = read_plan(plan_path)
    if not segments:
        raise EditorError(
            f"Plan {plan_path} has no segments. Re-run analyze with a lower scene threshold."
        )
    estimated_duration_seconds = _estimate_render_duration(
        segments,
        crossfade_seconds=crossfade_seconds,
    )

    include_audio = _has_audio_stream(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    crop_filter: str | None = None
    if auto_crop:
        try:
            duration_seconds = _probe_duration_seconds(input_path)
            crop_filter = _detect_watchability_crop_filter(
                input_path,
                duration_seconds=duration_seconds,
            )
        except (EditorError, subprocess.CalledProcessError):
            crop_filter = None

    if not include_audio:
        filter_graph, map_video, map_audio = _build_filter_complex(
            segments,
            include_audio=False,
            include_video=True,
            allow_upscale=allow_upscale,
            crop_filter=crop_filter,
            crossfade_seconds=crossfade_seconds,
            audio_fade_seconds=audio_fade_seconds,
        )
        if map_video is None:
            raise EditorError("Failed to build video filter graph.")
        command = [
            "ffmpeg",
            "-hide_banner",
            "-stats_period",
            "0.25",
            "-y",
            "-i",
            str(input_path),
            "-filter_complex",
            filter_graph,
            "-map",
            map_video,
            *_video_codec_args(video_encoder=video_encoder, crf=crf, preset=preset),
            "-movflags",
            "+faststart",
            "-an",
            str(output_path),
        ]
        _run_command(command)
        return

    audio_analysis_command: list[str] | None = None
    if two_pass_loudnorm:
        analysis_filter_graph, _, analysis_audio_map = _build_filter_complex(
            segments,
            include_audio=True,
            include_video=False,
            crossfade_seconds=crossfade_seconds,
            audio_fade_seconds=audio_fade_seconds,
            audio_post_filter=LOUDNORM_ANALYSIS_FILTER,
        )
        if analysis_audio_map is not None:
            audio_analysis_command = [
                "ffmpeg",
                "-hide_banner",
                "-y",
                "-i",
                str(input_path),
                "-filter_complex",
                analysis_filter_graph,
                "-map",
                analysis_audio_map,
                "-f",
                "null",
                "-",
            ]

    loudnorm_filter = _select_loudnorm_filter(
        two_pass_loudnorm=two_pass_loudnorm,
        analysis_command=audio_analysis_command,
    )

    normalized_filter_graph, map_video, map_audio = _build_filter_complex(
        segments,
        include_audio=True,
        include_video=True,
        allow_upscale=allow_upscale,
        crop_filter=crop_filter,
        crossfade_seconds=crossfade_seconds,
        audio_fade_seconds=audio_fade_seconds,
        audio_post_filter=loudnorm_filter,
    )
    if map_video is None or map_audio is None:
        raise EditorError("Failed to build audio/video filter graph.")

    fallback_filter_graph, fallback_video_map, fallback_audio_map = _build_filter_complex(
        segments,
        include_audio=True,
        include_video=True,
        allow_upscale=allow_upscale,
        crop_filter=crop_filter,
        crossfade_seconds=crossfade_seconds,
        audio_fade_seconds=audio_fade_seconds,
        audio_post_filter=None,
    )
    if fallback_video_map is None or fallback_audio_map is None:
        raise EditorError("Failed to build fallback audio/video filter graph.")

    normalized_command = [
        "ffmpeg",
        "-hide_banner",
        "-stats_period",
        "0.25",
        "-y",
        "-i",
        str(input_path),
        "-filter_complex",
        normalized_filter_graph,
        "-map",
        map_video,
        *_video_codec_args(video_encoder=video_encoder, crf=crf, preset=preset),
        "-movflags",
        "+faststart",
        "-map",
        map_audio,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ac",
        "2",
        "-ar",
        "48000",
        str(output_path),
    ]
    fallback_command = [
        "ffmpeg",
        "-hide_banner",
        "-stats_period",
        "0.25",
        "-y",
        "-i",
        str(input_path),
        "-filter_complex",
        fallback_filter_graph,
        "-map",
        fallback_video_map,
        *_video_codec_args(video_encoder=video_encoder, crf=crf, preset=preset),
        "-movflags",
        "+faststart",
        "-map",
        fallback_audio_map,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ac",
        "2",
        "-ar",
        "48000",
        str(output_path),
    ]
    _run_with_loudnorm_fallback(
        normalized_command,
        fallback_command,
        progress_label="Rendering highlights",
        progress_duration_seconds=estimated_duration_seconds,
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
    has_audio = _has_audio_stream(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    crop_filter: str | None = None
    duration_seconds: float | None = None
    try:
        duration_seconds = _probe_duration_seconds(input_path)
    except (EditorError, subprocess.CalledProcessError):
        duration_seconds = None
    if auto_crop and duration_seconds:
        try:
            crop_filter = _detect_watchability_crop_filter(
                input_path,
                duration_seconds=duration_seconds,
            )
        except (EditorError, subprocess.CalledProcessError):
            crop_filter = None

    base_command = [
        "ffmpeg",
        "-hide_banner",
        "-stats_period",
        "0.25",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        _video_postprocess_filter(allow_upscale=allow_upscale, crop_filter=crop_filter),
        *_video_codec_args(video_encoder=video_encoder, crf=crf, preset=preset),
        "-movflags",
        "+faststart",
    ]

    if not has_audio:
        command = base_command + ["-an", str(output_path)]
        _run_command(command)
        return

    analysis_command: list[str] | None = None
    if two_pass_loudnorm:
        analysis_command = [
            "ffmpeg",
            "-hide_banner",
            "-stats_period",
            "0.25",
            "-y",
            "-i",
            str(input_path),
            "-vn",
            "-af",
            LOUDNORM_ANALYSIS_FILTER,
            "-f",
            "null",
            "-",
        ]

    loudnorm_filter = _select_loudnorm_filter(
        two_pass_loudnorm=two_pass_loudnorm,
        analysis_command=analysis_command,
    )
    normalized_command = base_command + [
        "-af",
        loudnorm_filter,
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ac",
        "2",
        "-ar",
        "48000",
        str(output_path),
    ]
    fallback_command = base_command + [
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ac",
        "2",
        "-ar",
        "48000",
        str(output_path),
    ]
    _run_with_loudnorm_fallback(
        normalized_command,
        fallback_command,
        progress_label="Rendering full match",
        progress_duration_seconds=duration_seconds,
    )


def _build_thumbnail_filter(
    *,
    width: int,
    height: int,
    crop_filter: str | None,
    enhance: bool,
) -> str:
    filters: list[str] = []
    if crop_filter:
        filters.append(crop_filter)
    filters.extend(
        [
            f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
        ]
    )
    if enhance:
        filters.extend(
            [
                "eq=saturation=1.10:contrast=1.06:brightness=0.015",
                "unsharp=5:5:0.8:3:3:0.4",
            ]
        )
    filters.append("format=yuv420p")
    return ",".join(filters)


def _build_thumbnail_filter_complex(
    *,
    width: int,
    height: int,
    crop_filter: str | None,
    enhance: bool,
    champion_input_index: int | None,
    champion_scale: float,
    champion_anchor: str,
    headline_input_index: int | None,
) -> tuple[str, str]:
    chain_parts: list[str] = []
    if crop_filter:
        chain_parts.append(crop_filter)
    chain_parts.extend(
        [
            f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
        ]
    )
    if enhance:
        chain_parts.extend(
            [
                "eq=saturation=1.10:contrast=1.06:brightness=0.015",
                "unsharp=5:5:0.8:3:3:0.4",
            ]
        )
    chain_parts.append("format=yuv420p")

    lines = [f"[0:v]{','.join(chain_parts)}[thumb_base]"]
    current = "thumb_base"

    if champion_input_index is not None:
        max_overlay_width = max(96, int(round(width * champion_scale)))
        max_overlay_height = max(96, int(round(height * 0.96)))
        lines.append(
            f"[{champion_input_index}:v]"
            f"scale=w={max_overlay_width}:h={max_overlay_height}:force_original_aspect_ratio=decrease"
            "[champ]"
        )
        if champion_anchor == "left":
            x_expr = "24"
        elif champion_anchor == "center":
            x_expr = "(main_w-overlay_w)/2"
        else:
            x_expr = "main_w-overlay_w-24"
        y_expr = "main_h-overlay_h-14"
        lines.append(
            f"[{current}][champ]overlay=x={x_expr}:y={y_expr}:format=auto[thumb_overlay]"
        )
        current = "thumb_overlay"

    if headline_input_index is not None:
        lines.append(f"[{headline_input_index}:v]format=rgba[text]")
        lines.append(f"[{current}][text]overlay=x=0:y=0:format=auto[thumb_text]")
        current = "thumb_text"

    return ";".join(lines), f"[{current}]"


def _select_default_headline_font() -> Path | None:
    candidates = (
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Trebuchet MS Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists() and path.is_file():
            return path
    return None


def _create_headline_overlay_image(
    *,
    output_path: Path,
    width: int,
    height: int,
    headline_text: str,
    headline_size: int,
    headline_color: str,
    headline_font: Path | None,
    headline_y_ratio: float,
) -> None:
    try:
        from PIL import Image, ImageColor, ImageDraw, ImageFont
    except ImportError as error:
        raise EditorError(
            "Headline text overlay requires Pillow. Run: python3 -m pip install pillow"
        ) from error

    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    font_path = headline_font or _select_default_headline_font()
    font = None
    if font_path is not None:
        try:
            font = ImageFont.truetype(str(font_path), headline_size)
        except OSError:
            font = None
    if font is None:
        try:
            font = ImageFont.truetype("Arial.ttf", headline_size)
        except OSError:
            font = ImageFont.load_default()

    try:
        rgb = ImageColor.getrgb(headline_color.strip() or "white")
    except ValueError as error:
        raise EditorError(
            f"Invalid --headline-color value: {headline_color!r}"
        ) from error

    stroke_width = max(2, int(round(headline_size * 0.065)))
    line_spacing = max(4, int(round(headline_size * 0.16)))
    lines = [line for line in headline_text.splitlines() if line.strip()]
    if not lines:
        lines = [headline_text.strip()]

    measurements: list[tuple[str, int, int]] = []
    max_width = 0
    total_height = 0
    for index, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
        line_width = max(0, bbox[2] - bbox[0])
        line_height = max(0, bbox[3] - bbox[1])
        measurements.append((line, line_width, line_height))
        max_width = max(max_width, line_width)
        total_height += line_height
        if index + 1 < len(lines):
            total_height += line_spacing

    top = int(height * max(0.0, min(1.0, headline_y_ratio)))
    top = min(max(8, top), max(8, height - total_height - 16))

    y_cursor = top
    fill_color = (rgb[0], rgb[1], rgb[2], 255)
    for line, line_width, line_height in measurements:
        x = max(0, (width - line_width) // 2)
        draw.text(
            (x, y_cursor),
            line,
            font=font,
            fill=fill_color,
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0, 230),
        )
        y_cursor += line_height + line_spacing

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _select_thumbnail_timestamp(
    *,
    duration_seconds: float,
    vision_windows: list[VisionWindow],
) -> float:
    if duration_seconds <= 0:
        return 0.0

    safe_start = max(0.0, min(duration_seconds - 0.05, max(1.5, duration_seconds * 0.06)))
    safe_end = max(safe_start + 0.05, duration_seconds - max(1.5, duration_seconds * 0.04))
    fallback = min(safe_end, max(safe_start, duration_seconds * 0.42))

    if not vision_windows:
        return fallback

    candidates = [
        window
        for window in vision_windows
        if window.end > window.start and window.end >= safe_start and window.start <= safe_end
    ]
    if not candidates:
        candidates = [window for window in vision_windows if window.end > window.start]
    if not candidates:
        return fallback

    motions = [window.motion for window in candidates if math.isfinite(window.motion)]
    saturations = [window.saturation for window in candidates if math.isfinite(window.saturation)]
    densities = [window.scene_density for window in candidates if math.isfinite(window.scene_density)]
    motion_low = _percentile(motions, 0.15)
    motion_high = _percentile(motions, 0.90)
    sat_low = _percentile(saturations, 0.15)
    sat_high = _percentile(saturations, 0.90)
    density_low = _percentile(densities, 0.10)
    density_high = _percentile(densities, 0.90)

    best_score = -1.0
    best_timestamp = fallback
    for window in candidates:
        center = (window.start + window.end) / 2.0
        clamped_center = min(safe_end, max(safe_start, center))
        timeline_ratio = clamped_center / duration_seconds if duration_seconds > 0 else 0.5
        midpoint_bias = 1.0 - min(1.0, abs(timeline_ratio - 0.55) / 0.55)

        motion_norm = _normalize(window.motion, low=motion_low, high=motion_high)
        sat_norm = _normalize(window.saturation, low=sat_low, high=sat_high)
        density_norm = _normalize(window.scene_density, low=density_low, high=density_high)
        score = (
            0.62 * max(0.0, min(1.0, window.score))
            + 0.14 * motion_norm
            + 0.10 * density_norm
            + 0.10 * sat_norm
            + 0.04 * midpoint_bias
        )
        if score > best_score:
            best_score = score
            best_timestamp = clamped_center

    return best_timestamp


def generate_thumbnail(
    *,
    input_path: Path,
    output_path: Path,
    timestamp_seconds: float | None,
    scene_threshold: float,
    vision_sample_fps: float,
    vision_window_seconds: float,
    vision_step_seconds: float,
    width: int,
    height: int,
    quality: int,
    auto_crop: bool,
    enhance: bool,
    champion_overlay_path: Path | None,
    champion_scale: float,
    champion_anchor: str,
    headline_text: str | None,
    headline_size: int,
    headline_color: str,
    headline_font: Path | None,
    headline_y_ratio: float,
) -> dict[str, float | int | bool]:
    duration_seconds = _probe_duration_seconds(input_path)
    auto_selected = timestamp_seconds is None
    scene_threshold_used = scene_threshold
    events: list[float] = []
    vision_windows: list[VisionWindow] = []

    if auto_selected:
        events, scene_threshold_used = detect_scene_events_adaptive(
            input_path,
            scene_threshold,
            duration_seconds=duration_seconds,
            show_progress=True,
            progress_label="Analyzing scenes",
        )
        vision_windows = score_vision_activity(
            input_path,
            events=events,
            duration_seconds=duration_seconds,
            sample_fps=vision_sample_fps,
            window_seconds=vision_window_seconds,
            step_seconds=vision_step_seconds,
            show_progress=True,
            progress_label="Scoring gameplay",
        )
        selected_timestamp = _select_thumbnail_timestamp(
            duration_seconds=duration_seconds,
            vision_windows=vision_windows,
        )
    else:
        selected_timestamp = float(timestamp_seconds)

    if not math.isfinite(selected_timestamp):
        selected_timestamp = 0.0
    if duration_seconds > 0:
        selected_timestamp = min(max(0.0, selected_timestamp), max(0.0, duration_seconds - 0.05))

    crop_filter: str | None = None
    if auto_crop and duration_seconds > 0:
        try:
            crop_filter = _detect_watchability_crop_filter(
                input_path,
                duration_seconds=duration_seconds,
            )
        except (EditorError, subprocess.CalledProcessError):
            crop_filter = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_headline = None
    if headline_text is not None and headline_text.strip():
        normalized_headline = headline_text.replace("\\n", "\n").strip()

    with tempfile.TemporaryDirectory(prefix="lol-thumb-") as temp_dir:
        headline_path: Path | None = None
        if normalized_headline:
            headline_path = Path(temp_dir) / "headline_overlay.png"
            _create_headline_overlay_image(
                output_path=headline_path,
                width=width,
                height=height,
                headline_text=normalized_headline,
                headline_size=headline_size,
                headline_color=headline_color,
                headline_font=headline_font,
                headline_y_ratio=headline_y_ratio,
            )

        champion_input_index: int | None = None
        headline_input_index: int | None = None
        next_input_index = 1
        if champion_overlay_path is not None:
            champion_input_index = next_input_index
            next_input_index += 1
        if headline_path is not None:
            headline_input_index = next_input_index

        filter_graph, output_label = _build_thumbnail_filter_complex(
            width=width,
            height=height,
            crop_filter=crop_filter,
            enhance=enhance,
            champion_input_index=champion_input_index,
            champion_scale=champion_scale,
            champion_anchor=champion_anchor,
            headline_input_index=headline_input_index,
        )
        command = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-ss",
            f"{selected_timestamp:.3f}",
            "-i",
            str(input_path),
        ]
        if champion_overlay_path is not None:
            command.extend(["-i", str(champion_overlay_path)])
        if headline_path is not None:
            command.extend(["-i", str(headline_path)])
        command.extend(
            [
                "-frames:v",
                "1",
                "-filter_complex",
                filter_graph,
                "-map",
                output_label,
            ]
        )
        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
            command.extend(["-q:v", str(quality)])
        command.append(str(output_path))
        _run_command(command, capture_output=True)

    return {
        "timestamp_seconds": round(selected_timestamp, 3),
        "duration_seconds": round(duration_seconds, 3),
        "auto_selected": auto_selected,
        "scene_threshold_used": round(scene_threshold_used, 4),
        "event_count": len(events),
        "vision_window_count": len(vision_windows),
        "used_champion_overlay": champion_overlay_path is not None,
        "used_headline_text": normalized_headline is not None,
    }


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
    scores = [window.score for window in vision_windows]
    if scores:
        high_threshold = _percentile(scores, 0.74)
        low_threshold = _percentile(scores, 0.32)
        high_activity_ratio = sum(1 for score in scores if score >= high_threshold) / len(scores)
        low_activity_ratio = sum(1 for score in scores if score <= low_threshold) / len(scores)
    else:
        high_activity_ratio = 0.0
        low_activity_ratio = 1.0

    tuned_clip_before = clip_before
    tuned_clip_after = clip_after
    tuned_gap = min_gap_seconds
    tuned_max_clips = max_clips
    tuned_forced_share = forced_cue_share

    # Preserve more setup/payoff in action-heavy matches and keep more forced kill/objective cues.
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
) -> dict[str, int | float | bool]:
    source_duration_seconds = _probe_duration_seconds(input_path)
    duration_seconds = source_duration_seconds
    terminal_result_seconds: float | None = None
    if end_on_result:
        try:
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
    events, scene_threshold_used = detect_scene_events_adaptive(
        input_path,
        scene_threshold,
        duration_seconds=duration_seconds,
        show_progress=True,
    )
    events = [event for event in events if 0.0 <= event <= duration_seconds]
    resolved_target_duration_seconds = target_duration_seconds
    if resolved_target_duration_seconds <= 0 and target_duration_ratio > 0:
        resolved_target_duration_seconds = duration_seconds * min(1.0, target_duration_ratio)
    vision_windows: list[VisionWindow] = []
    if vision_scoring != "off":
        try:
            vision_windows = score_vision_activity(
                input_path,
                events=events,
                duration_seconds=duration_seconds,
                sample_fps=vision_sample_fps,
                window_seconds=vision_window_seconds,
                step_seconds=vision_step_seconds,
                show_progress=True,
            )
        except subprocess.CalledProcessError as error:
            print(
                "Warning: vision scoring failed; falling back to scene-only scoring.",
                file=sys.stderr,
            )
            if error.stderr:
                print(error.stderr, file=sys.stderr)
        vision_windows = [window for window in vision_windows if window.end <= duration_seconds]

    ai_cues: list[TranscriptionCue] = []
    whisper_cues: list[TranscriptionCue] = []
    ocr_cues: list[TranscriptionCue] = []
    ai_cue_threshold_used = ai_cue_threshold
    if vision_scoring == "local-ai":
        if whisper_model is None:
            raise EditorError(
                "--whisper-model is required when --vision-scoring local-ai."
            )
        print("Running local AI transcription (whisper.cpp)...", file=sys.stderr, flush=True)
        resolved_whisper_vad = whisper_vad
        if whisper_vad and whisper_vad_model is None:
            resolved_whisper_vad = False
            print(
                (
                    "Warning: --whisper-vad requested but no --whisper-vad-model was provided; "
                    "continuing without VAD."
                ),
                file=sys.stderr,
                flush=True,
            )
        try:
            raw_ai_cues = _collect_local_ai_cues(
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
            raw_ai_cues = [cue for cue in raw_ai_cues if cue.end <= duration_seconds]
        except subprocess.CalledProcessError as error:
            stderr_summary = (error.stderr or "").strip()
            detail = f"\n{stderr_summary}" if stderr_summary else ""
            raise EditorError(
                "Local AI transcription failed while running whisper.cpp."
                f"{detail}"
            ) from error
        thresholds = _adaptive_ai_cue_thresholds(ai_cue_threshold)
        for candidate_threshold in thresholds:
            candidate_cues = [cue for cue in raw_ai_cues if cue.score >= candidate_threshold]
            if candidate_cues:
                whisper_cues = candidate_cues
                ai_cue_threshold_used = candidate_threshold
                break
        if whisper_cues and ai_cue_threshold_used < ai_cue_threshold:
            print(
                (
                    f"Local AI cue threshold adjusted to {ai_cue_threshold_used:.2f}; "
                    f"detected {len(whisper_cues)} transcript cues."
                ),
                file=sys.stderr,
                flush=True,
            )

        if ocr_cue_scoring != "off":
            print("Running local OCR cue detection (tesseract)...", file=sys.stderr, flush=True)
            try:
                ocr_cues = _collect_ocr_cues(
                    input_path=input_path,
                    tesseract_binary=tesseract_bin,
                    sample_fps=ocr_sample_fps,
                    cue_threshold=ocr_cue_threshold,
                )
                ocr_cues = [cue for cue in ocr_cues if cue.end <= duration_seconds]
                if ocr_cues:
                    print(
                        f"Detected {len(ocr_cues)} OCR cues from HUD text.",
                        file=sys.stderr,
                        flush=True,
                    )
            except EditorError as error:
                print(
                    f"Warning: OCR cue detection unavailable ({error}); continuing without OCR cues.",
                    file=sys.stderr,
                    flush=True,
                )
            except subprocess.CalledProcessError as error:
                stderr_summary = (error.stderr or "").strip()
                detail = f" ({stderr_summary})" if stderr_summary else ""
                print(
                    f"Warning: OCR cue detection failed{detail}; continuing without OCR cues.",
                    file=sys.stderr,
                    flush=True,
                )

        combined_cues = sorted(whisper_cues + ocr_cues, key=lambda cue: (cue.start, -cue.score))
        for cue in combined_cues:
            if ai_cues and cue.start - ai_cues[-1].start < 8.0:
                if cue.score > ai_cues[-1].score:
                    ai_cues[-1] = cue
                continue
            ai_cues.append(cue)

        if not ai_cues:
            print(
                "Local AI did not detect transcript/OCR keyword cues; using vision/scene scoring only.",
                file=sys.stderr,
                flush=True,
            )

    ai_priority_cues = _extract_ai_priority_cues(ai_cues)
    if ai_priority_cues:
        events = sorted(events + ai_priority_cues)
        vision_windows = _boost_vision_windows_with_ai_cues(vision_windows, ai_cues)

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
                (
                    "One-shot smart tuning: "
                    f"clip-before={effective_clip_before:.1f}, "
                    f"clip-after={effective_clip_after:.1f}, "
                    f"min-gap={effective_min_gap_seconds:.1f}, "
                    f"max-clips={effective_max_clips}, "
                    f"forced-cue-share={effective_forced_cue_share:.2f}"
                ),
                file=sys.stderr,
                flush=True,
            )

    segments, used_fallback = build_segments(
        events,
        duration_seconds=duration_seconds,
        clip_before=effective_clip_before,
        clip_after=effective_clip_after,
        min_gap_seconds=effective_min_gap_seconds,
        max_clips=effective_max_clips,
        target_duration_seconds=resolved_target_duration_seconds,
        intro_seconds=intro_seconds,
        outro_seconds=outro_seconds,
        vision_windows=vision_windows,
        ai_priority_cues=ai_priority_cues,
        ai_priority_details=ai_cues,
        force_outro_to_duration_end=terminal_result_seconds is not None,
        forced_cue_share=effective_forced_cue_share,
    )
    settings = {
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
        "target_duration_seconds": resolved_target_duration_seconds,
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
        "whisper_vad_model": str(whisper_vad_model) if whisper_vad_model is not None else "",
        "ocr_cue_scoring": ocr_cue_scoring,
        "tesseract_bin": tesseract_bin,
        "ocr_sample_fps": ocr_sample_fps,
        "ocr_cue_threshold": ocr_cue_threshold,
        "ai_cue_threshold": ai_cue_threshold,
        "ai_cue_threshold_used": ai_cue_threshold_used,
    }
    write_plan(
        input_path=input_path,
        output_path=plan_path,
        duration_seconds=duration_seconds,
        events=events,
        segments=segments,
        used_fallback=used_fallback,
        settings=settings,
        ai_cues=ai_cues,
    )
    return {
        "event_count": len(events),
        "segment_count": len(segments),
        "used_fallback": used_fallback,
        "duration_seconds": duration_seconds,
        "source_duration_seconds": source_duration_seconds,
        "terminal_result_seconds": (
            round(terminal_result_seconds, 3) if terminal_result_seconds is not None else 0.0
        ),
        "vision_window_count": len(vision_windows),
        "scene_threshold_used": scene_threshold_used,
        "ai_cue_count": len(ai_cues),
        "whisper_cue_count": len(whisper_cues),
        "ocr_cue_count": len(ocr_cues),
    }


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _average(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(0.0, variance))


def _closeness_score(value: float, *, target: float, tolerance: float) -> float:
    if tolerance <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(value - target) / tolerance)


def _build_watchability_report(
    *,
    duration_seconds: float,
    events: list[float],
    vision_windows: list[VisionWindow],
    scene_threshold_used: float,
) -> dict[str, object]:
    duration_minutes = duration_seconds / 60 if duration_seconds > 0 else 0.0
    event_rate_per_minute = len(events) / duration_minutes if duration_minutes > 0 else 0.0

    scores = [window.score for window in vision_windows]
    saturations = [window.saturation for window in vision_windows]
    avg_score = _average(scores)
    score_std = _stddev(scores)
    low_activity_threshold = _percentile(scores, 0.30)
    high_activity_threshold = _percentile(scores, 0.75)
    low_activity_ratio = (
        sum(1 for score in scores if score <= low_activity_threshold) / len(scores)
        if scores
        else 1.0
    )
    high_activity_ratio = (
        sum(1 for score in scores if score >= high_activity_threshold) / len(scores)
        if scores
        else 0.0
    )
    ordered_windows = sorted(vision_windows, key=lambda window: window.start)
    death_cues = _detect_death_cues(ordered_windows, min_spacing_seconds=20.0)
    motions = [window.motion for window in ordered_windows]
    low_motion_threshold = _percentile(motions, 0.45)
    low_score_threshold = _percentile(scores, 0.50)
    low_saturation_threshold = _percentile(saturations, 0.30)
    avg_window_duration = _average([window.end - window.start for window in ordered_windows])
    death_context_seconds = max(12.0, avg_window_duration * 1.5)
    death_like_windows = 0
    for window in ordered_windows:
        if not death_cues:
            break
        center = (window.start + window.end) / 2
        near_death_transition = any(
            abs(center - cue_time) <= death_context_seconds for cue_time in death_cues
        )
        if not near_death_transition:
            continue
        if (
            window.motion <= low_motion_threshold
            and window.score <= low_score_threshold
            and window.saturation <= low_saturation_threshold
        ):
            death_like_windows += 1
    gray_ratio = (
        death_like_windows / len(ordered_windows)
        if ordered_windows
        else 0.0
    )

    expected_event_rate_per_minute = min(
        14.0,
        max(0.8, 0.9 + 7.5 * avg_score + 3.0 * high_activity_ratio),
    )
    if avg_score < 0.34 and low_activity_ratio > 0.50:
        activity_profile = "low-action"
        expected_event_rate_per_minute = min(expected_event_rate_per_minute, 3.0)
    elif avg_score > 0.62 or high_activity_ratio > 0.30:
        activity_profile = "high-action"
    else:
        activity_profile = "standard"

    pacing_tolerance = max(2.0, expected_event_rate_per_minute * 0.9)
    raw_pacing = _closeness_score(
        event_rate_per_minute,
        target=expected_event_rate_per_minute,
        tolerance=pacing_tolerance,
    )
    pacing_floor = 0.35 if activity_profile == "low-action" else 0.18
    engagement_component = avg_score
    pacing_component = max(pacing_floor, raw_pacing)
    variety_component = min(1.0, score_std / 0.18)
    idle_component = 1.0 - low_activity_ratio
    gray_penalty_scale = 0.55 if activity_profile == "low-action" else 1.0
    gray_component = 1.0 - min(1.0, gray_ratio * 0.9 * gray_penalty_scale)

    overall_score = 100.0 * (
        0.35 * engagement_component
        + 0.15 * pacing_component
        + 0.22 * variety_component
        + 0.23 * idle_component
        + 0.05 * gray_component
    )
    overall_score = min(100.0, max(0.0, overall_score))

    if overall_score >= 80:
        rating = "excellent"
    elif overall_score >= 65:
        rating = "good"
    elif overall_score >= 50:
        rating = "fair"
    else:
        rating = "rough"

    quality_target_event_rate = {
        "low-action": 1.6,
        "standard": 2.5,
        "high-action": 3.6,
    }[activity_profile]
    quality_tolerance = max(0.9, quality_target_event_rate * 0.55)
    quality_pacing_raw = _closeness_score(
        event_rate_per_minute,
        target=quality_target_event_rate,
        tolerance=quality_tolerance,
    )
    quality_pacing_floor = 0.30 if activity_profile == "low-action" else 0.15
    quality_pacing = max(quality_pacing_floor, quality_pacing_raw)
    quality_variety = min(1.0, score_std / 0.16)
    quality_intensity = min(1.0, max(0.0, avg_score))
    quality_focus = 1.0 - min(1.0, gray_ratio * (0.55 if activity_profile == "low-action" else 0.95))
    quality_dead_air = 1.0 - low_activity_ratio
    quality_action_mix = min(1.0, max(0.0, high_activity_ratio / 0.45))
    highlight_quality_score = 100.0 * (
        0.27 * quality_intensity
        + 0.24 * quality_pacing
        + 0.19 * quality_variety
        + 0.14 * quality_action_mix
        + 0.11 * quality_dead_air
        + 0.05 * quality_focus
    )
    highlight_quality_score = min(100.0, max(0.0, highlight_quality_score))
    if highlight_quality_score >= 80:
        quality_rating = "excellent"
    elif highlight_quality_score >= 65:
        quality_rating = "good"
    elif highlight_quality_score >= 50:
        quality_rating = "fair"
    else:
        quality_rating = "rough"

    if activity_profile == "low-action":
        watchability_weight = 0.30
        quality_weight = 0.70
    elif activity_profile == "high-action":
        watchability_weight = 0.55
        quality_weight = 0.45
    else:
        watchability_weight = 0.45
        quality_weight = 0.55
    youtube_score = (
        overall_score * watchability_weight + highlight_quality_score * quality_weight
    )
    youtube_score = min(100.0, max(0.0, youtube_score))

    recommendations: list[str] = []
    if low_activity_ratio > 0.45:
        recommendations.append("Trim or speed up extended low-activity windows.")
    if event_rate_per_minute > 18:
        recommendations.append("Reduce cut frequency to improve flow continuity.")
    low_pace_threshold = max(1.2, expected_event_rate_per_minute * 0.45)
    if event_rate_per_minute < low_pace_threshold:
        if activity_profile == "low-action":
            recommendations.append(
                "This match looks naturally low-action; prioritize narrative moments over forcing extra cuts."
            )
        else:
            recommendations.append("Include more high-impact moments (fights, objectives, deaths).")
            recommendations.append(
                "Re-analyze with --max-clips 24 and --min-gap-seconds 14 to include more highlight moments."
            )
    if high_activity_ratio < 0.18:
        recommendations.append("Prioritize more high-intensity sequences to improve excitement.")
    if gray_ratio > 0.30 and activity_profile != "low-action":
        recommendations.append(
            "Death/gray-screen time is high; keep only the most informative examples."
        )
    if gray_ratio > 0.30 and activity_profile == "low-action":
        recommendations.append(
            "Gray-screen sections are acceptable if spectating adds context; trim only stagnant portions."
        )

    quality_recommendations: list[str] = []
    if quality_pacing_raw < 0.45:
        if event_rate_per_minute < quality_target_event_rate:
            quality_recommendations.append(
                "General quality: tighten setup/payoff windows around key moments to improve momentum."
            )
        else:
            quality_recommendations.append(
                "General quality: add more context around major moments to avoid over-cut pacing."
            )
    if quality_variety < 0.42:
        quality_recommendations.append(
            "General quality: increase moment variety (teamfights, objectives, picks, pushes, clutch escapes)."
        )
    if quality_intensity < 0.42 and high_activity_ratio < 0.20:
        quality_recommendations.append(
            "General quality: bias selection more toward the highest-intensity windows."
        )
    if gray_ratio > 0.25 and activity_profile != "low-action":
        quality_recommendations.append(
            "General quality: trim low-information death/spectator downtime unless it adds strategic context."
        )

    merged_recommendations = recommendations + [
        recommendation
        for recommendation in quality_recommendations
        if recommendation not in recommendations
    ]

    return {
        "watchability_score": round(overall_score, 2),
        "rating": rating,
        "highlight_quality_score": round(highlight_quality_score, 2),
        "quality_rating": quality_rating,
        "youtube_score": round(youtube_score, 2),
        "score_blend": {
            "watchability_weight": round(watchability_weight, 2),
            "quality_weight": round(quality_weight, 2),
        },
        "activity_profile": activity_profile,
        "scene_threshold_used": round(scene_threshold_used, 4),
        "duration_seconds": round(duration_seconds, 3),
        "event_count": len(events),
        "event_rate_per_minute": round(event_rate_per_minute, 3),
        "expected_event_rate_per_minute": round(expected_event_rate_per_minute, 3),
        "avg_activity_score": round(avg_score, 4),
        "activity_score_stddev": round(score_std, 4),
        "low_activity_ratio": round(low_activity_ratio, 4),
        "high_activity_ratio": round(high_activity_ratio, 4),
        "gray_screen_ratio": round(gray_ratio, 4),
        "quality_recommendations": quality_recommendations,
        "recommendations": merged_recommendations,
    }


def analyze_watchability(
    *,
    input_path: Path,
    scene_threshold: float,
    vision_sample_fps: float,
    vision_window_seconds: float,
    vision_step_seconds: float,
    show_progress: bool = True,
) -> dict[str, object]:
    last_progress_ratio = -1.0

    def emit_progress(ratio: float) -> None:
        nonlocal last_progress_ratio
        clamped_ratio = min(1.0, max(0.0, ratio))
        if clamped_ratio < last_progress_ratio + 0.01 and clamped_ratio < 1.0:
            return
        if clamped_ratio >= 1.0:
            print(
                _render_progress_line("Analyzing watchability", clamped_ratio),
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                _render_progress_line("Analyzing watchability", clamped_ratio),
                end="",
                file=sys.stderr,
                flush=True,
            )
        last_progress_ratio = clamped_ratio

    def stage_progress_callback(start_ratio: float, end_ratio: float) -> Callable[[float], None]:
        stage_span = max(0.0, end_ratio - start_ratio)

        def update(stage_ratio: float) -> None:
            emit_progress(start_ratio + stage_span * min(1.0, max(0.0, stage_ratio)))

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
        progress_callback=(
            stage_progress_callback(0.05, 0.55) if show_progress else None
        ),
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
        progress_callback=(
            stage_progress_callback(0.55, 0.95) if show_progress else None
        ),
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


def _format_youtube_timestamp(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _keyword_to_hashtag(keyword: str) -> str:
    pieces = [piece for piece in re.split(r"[^a-zA-Z0-9]+", keyword.strip()) if piece]
    if not pieces:
        return ""
    return "#" + "".join(piece[:1].upper() + piece[1:] for piece in pieces)


def _cue_label(cue: TranscriptionCue) -> str:
    keywords = set(cue.keywords)
    if "pentakill" in keywords:
        return "Pentakill spike"
    if "quadra kill" in keywords:
        return "Quadra kill play"
    if "triple kill" in keywords:
        return "Triple kill sequence"
    if "shutdown" in keywords:
        return "Shutdown swing"
    if "ace" in keywords:
        return "Ace teamfight"
    if {"baron", "baron nashor", "nashor"} & keywords:
        return "Baron objective fight"
    if {"elder dragon", "dragon"} & keywords:
        return "Dragon objective fight"
    if {"nexus", "victory", "defeat"} & keywords:
        return "Game-ending push"
    if {"turret", "tower", "inhibitor", "structure destroyed"} & keywords:
        return "Objective pressure"
    if {"teamfight", "fight"} & keywords:
        return "Heavy teamfight"
    if {"kill", "slain", "enemy slain"} & keywords:
        return "Pick and kill"
    return "High-impact moment"


def _sanitize_blurb(text: str, *, max_chars: int = 88) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    compact = compact.replace("[", "").replace("]", "")
    if len(compact) <= max_chars:
        return compact
    truncated = compact[: max_chars - 1].rstrip()
    return f"{truncated}…"


def _clean_moment_blurb(text: str, *, max_chars: int = 88) -> str | None:
    compact = _sanitize_blurb(text, max_chars=max_chars)
    if len(compact) < 16:
        return None
    alpha_count = sum(1 for char in compact if char.isalpha())
    alnum_count = sum(1 for char in compact if char.isalnum())
    if alnum_count <= 0:
        return None
    alpha_ratio = alpha_count / max(1, len(compact))
    if alpha_ratio < 0.48:
        return None
    words = re.findall(r"[A-Za-z]{3,}", compact)
    if len(words) < 3:
        return None
    return compact


def _dedupe_ranked_cues(
    cues: list[TranscriptionCue],
    *,
    min_gap_seconds: float = 10.0,
) -> list[TranscriptionCue]:
    ranked = sorted(cues, key=lambda cue: (-cue.score, cue.start))
    selected: list[TranscriptionCue] = []
    for cue in ranked:
        center = cue.center
        if any(abs(center - existing.center) < min_gap_seconds for existing in selected):
            continue
        selected.append(cue)
    return sorted(selected, key=lambda cue: cue.start)


def _fallback_moment_points_from_vision(
    *,
    vision_windows: list[VisionWindow],
    max_moments: int,
) -> list[tuple[float, str, str]]:
    if max_moments <= 0:
        return []
    ranked = sorted(vision_windows, key=lambda window: (-window.score, window.start))
    selected: list[tuple[float, str, str]] = []
    for window in ranked:
        center = (window.start + window.end) / 2.0
        if any(abs(center - existing[0]) < 14.0 for existing in selected):
            continue
        selected.append(
            (
                center,
                "High-intensity sequence",
                f"Activity score {window.score:.2f} (motion={window.motion:.2f}).",
            )
        )
        if len(selected) >= max_moments:
            break
    return sorted(selected, key=lambda item: item[0])


def _extract_moment_points(
    *,
    cues: list[TranscriptionCue],
    vision_windows: list[VisionWindow],
    max_moments: int,
) -> list[tuple[float, str, str]]:
    if max_moments <= 0:
        return []
    if not cues:
        return _fallback_moment_points_from_vision(
            vision_windows=vision_windows,
            max_moments=max_moments,
        )

    selected: list[tuple[float, str, str]] = []
    ranked = sorted(cues, key=lambda cue: (-cue.score, cue.start))
    for cue in ranked:
        center = cue.center
        if any(abs(center - existing[0]) < 12.0 for existing in selected):
            continue
        cleaned_blurb = _clean_moment_blurb(cue.text)
        if cleaned_blurb is None:
            cleaned_blurb = f"{_cue_label(cue)} momentum swing."
        selected.append(
            (
                center,
                _cue_label(cue),
                cleaned_blurb,
            )
        )
        if len(selected) >= max_moments:
            break
    return sorted(selected, key=lambda item: item[0])


def _keyword_weights_from_cues(cues: list[TranscriptionCue]) -> dict[str, float]:
    weighted = Counter()
    for cue in cues:
        unique_keywords = set(cue.keywords)
        if not unique_keywords:
            continue
        for keyword in unique_keywords:
            weighted[keyword] += cue.score
    return dict(weighted)


def _topic_from_keywords(
    *,
    keyword_weights: dict[str, float],
    activity_profile: str,
) -> str:
    suppressed_keywords = {
        "slain",
        "kill",
        "enemy slain",
        "ally slain",
        "fight",
        "teamfight",
        "flash",
        "ultimate",
        "ult",
    }
    prioritized = {
        keyword: weight
        for keyword, weight in keyword_weights.items()
        if keyword not in suppressed_keywords
    }

    if prioritized:
        top_keyword = sorted(
            prioritized.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
    elif keyword_weights:
        top_keyword = sorted(
            keyword_weights.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
    else:
        top_keyword = ""
    topic_map = {
        "pentakill": "PENTAKILL MOMENTS",
        "quadra kill": "QUADRA KILL CLUTCHES",
        "triple kill": "TRIPLE KILL SEQUENCES",
        "shutdown": "SHUTDOWN SWINGS",
        "ace": "ACE TEAMFIGHTS",
        "baron": "BARON FIGHTS",
        "baron nashor": "BARON NASHOR FIGHTS",
        "nashor": "BARON FIGHTS",
        "elder dragon": "ELDER DRAGON FIGHTS",
        "dragon": "DRAGON FIGHTS",
        "nexus": "GAME-ENDING PUSHES",
        "turret": "OBJECTIVE PUSHES",
        "tower": "OBJECTIVE PUSHES",
        "inhibitor": "OBJECTIVE PUSHES",
    }
    if top_keyword in topic_map:
        return topic_map[top_keyword]
    if top_keyword:
        return top_keyword.upper()
    if activity_profile == "high-action":
        return "INSANE TEAMFIGHTS"
    if activity_profile == "low-action":
        return "CLUTCH MACRO MOMENTS"
    return "RANKED HIGHLIGHTS"


def _build_ctr_title_candidates(
    *,
    champion: str,
    topic: str,
    youtube_score: float,
    moment_count: int,
    title_count: int,
) -> list[str]:
    base = champion.strip() if champion.strip() else "LoL"
    intensity = "INSANE" if youtube_score >= 75 else "CLUTCH" if youtube_score >= 62 else "HIGHLIGHT"
    candidates = [
        f"{base.upper()} {intensity}: {topic} | League of Legends Highlights",
        f"{moment_count} CRAZY MOMENTS - {base.upper()} {topic}",
        f"{base.upper()} HARD CARRY? {topic} (LoL Ranked)",
        f"{base.upper()} MONTAGE: {topic} | Ranked Gameplay",
        f"{topic} with {base.upper()} - League Highlights",
    ]
    deduped: list[str] = []
    for candidate in candidates:
        text = re.sub(r"\s+", " ", candidate).strip()
        if text in deduped:
            continue
        deduped.append(text)
        if len(deduped) >= max(1, title_count):
            break
    return deduped


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
    max_moments: int,
    title_count: int,
) -> dict[str, object]:
    duration_seconds = _probe_duration_seconds(input_path)
    events, scene_threshold_used = detect_scene_events_adaptive(
        input_path,
        scene_threshold,
        duration_seconds=duration_seconds,
        show_progress=True,
        progress_label="Analyzing scenes",
    )
    vision_windows = score_vision_activity(
        input_path,
        events=events,
        duration_seconds=duration_seconds,
        sample_fps=vision_sample_fps,
        window_seconds=vision_window_seconds,
        step_seconds=vision_step_seconds,
        show_progress=True,
        progress_label="Scoring gameplay",
    )
    watchability_report = _build_watchability_report(
        duration_seconds=duration_seconds,
        events=events,
        vision_windows=vision_windows,
        scene_threshold_used=scene_threshold_used,
    )

    whisper_cues: list[TranscriptionCue] = []
    if whisper_model is not None:
        print("Running local AI transcription (whisper.cpp)...", file=sys.stderr, flush=True)
        try:
            whisper_cues = _collect_local_ai_cues(
                input_path=input_path,
                whisper_model=whisper_model,
                whisper_binary=whisper_bin,
                whisper_language=whisper_language,
                whisper_threads=whisper_threads,
                cue_threshold=ai_cue_threshold,
                whisper_audio_stream=(
                    whisper_audio_stream if whisper_audio_stream >= 0 else None
                ),
                whisper_vad=whisper_vad,
                whisper_vad_threshold=whisper_vad_threshold,
                whisper_vad_model=whisper_vad_model,
            )
        except (EditorError, subprocess.CalledProcessError) as error:
            print(
                f"Warning: local AI transcription unavailable ({error}).",
                file=sys.stderr,
                flush=True,
            )

    ocr_cues: list[TranscriptionCue] = []
    if ocr_cue_scoring == "auto":
        print("Running OCR cue detection (tesseract)...", file=sys.stderr, flush=True)
        try:
            ocr_cues = _collect_ocr_cues(
                input_path=input_path,
                tesseract_binary=tesseract_bin,
                sample_fps=ocr_sample_fps,
                cue_threshold=ocr_cue_threshold,
            )
        except (EditorError, subprocess.CalledProcessError) as error:
            print(
                f"Warning: OCR cue detection unavailable ({error}).",
                file=sys.stderr,
                flush=True,
            )

    combined_cues = _dedupe_ranked_cues(whisper_cues + ocr_cues, min_gap_seconds=8.0)
    moment_points = _extract_moment_points(
        cues=combined_cues,
        vision_windows=vision_windows,
        max_moments=max_moments,
    )
    keyword_weights = _keyword_weights_from_cues(combined_cues)
    activity_profile = str(watchability_report.get("activity_profile", "standard"))
    topic = _topic_from_keywords(
        keyword_weights=keyword_weights,
        activity_profile=activity_profile,
    )

    youtube_score = float(watchability_report.get("youtube_score", 0.0))
    titles = _build_ctr_title_candidates(
        champion=champion,
        topic=topic,
        youtube_score=youtube_score,
        moment_count=len(moment_points),
        title_count=title_count,
    )

    chapter_points: list[tuple[float, str]] = [(0.0, "Opening setup")]
    chapter_points.extend((moment[0], moment[1]) for moment in moment_points if moment[0] >= 3.0)
    if duration_seconds >= 120:
        chapter_points.append(
            (
                max(0.0, duration_seconds - min(45.0, duration_seconds * 0.08)),
                "Final result / ending",
            )
        )
    chapter_points.sort(key=lambda item: item[0])
    deduped_chapters: list[tuple[float, str]] = []
    for chapter_time, chapter_label in chapter_points:
        if deduped_chapters and abs(chapter_time - deduped_chapters[-1][0]) < 8.0:
            continue
        deduped_chapters.append((chapter_time, chapter_label))
    chapters = [
        {"timestamp": _format_youtube_timestamp(chapter_time), "label": chapter_label}
        for chapter_time, chapter_label in deduped_chapters
    ]

    hashtags = ["#LeagueOfLegends", "#LoLHighlights", "#Gaming"]
    if champion.strip():
        champion_tag = _keyword_to_hashtag(champion)
        if champion_tag and champion_tag not in hashtags:
            hashtags.append(champion_tag)
    hashtag_suppressed = {
        "slain",
        "kill",
        "enemy slain",
        "ally slain",
        "fight",
        "teamfight",
        "flash",
        "ultimate",
        "ult",
    }
    for keyword, _ in sorted(keyword_weights.items(), key=lambda item: (-item[1], item[0])):
        if keyword in hashtag_suppressed:
            continue
        hashtag = _keyword_to_hashtag(keyword)
        if hashtag and hashtag not in hashtags:
            hashtags.append(hashtag)
        if len(hashtags) >= 9:
            break

    headline = titles[0] if titles else "League of Legends Ranked Highlights"
    description_lines: list[str] = [
        headline,
        "",
        (
            f"High-impact League highlights featuring {topic.lower()}, "
            f"{len(moment_points)} key moments, and {len(events)} major scene spikes."
        ),
        (
            "AI pacing score: "
            f"{watchability_report.get('youtube_score')}/100 "
            f"(quality={watchability_report.get('highlight_quality_score')}/100)."
        ),
        "",
        "What happens in this video:",
    ]
    if moment_points:
        for moment_time, moment_label, blurb in moment_points[:6]:
            description_lines.append(
                f"- {_format_youtube_timestamp(moment_time)} {moment_label}: {blurb}"
            )
    else:
        description_lines.append("- Fast-paced ranked highlights from this match.")

    description_lines.extend(
        [
            "",
            "Chapters:",
        ]
    )
    for chapter in chapters:
        description_lines.append(f"{chapter['timestamp']} {chapter['label']}")

    description_lines.extend(
        [
            "",
            (
                f"Subscribe for more ranked highlights on {channel_name}."
                if channel_name.strip()
                else "Subscribe for more ranked League highlight edits."
            ),
            "Like + comment your favorite moment so I can tune future edits.",
            "",
            " ".join(hashtags),
        ]
    )
    description_text = "\n".join(description_lines).strip()

    output_lines: list[str] = ["Title Ideas:"]
    for index, title in enumerate(titles, start=1):
        output_lines.append(f"{index}. {title}")
    output_lines.extend(["", "Description:", description_text])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines).strip() + "\n", encoding="utf-8")

    return {
        "titles": titles,
        "description_text": description_text,
        "hashtags": hashtags,
        "chapters": chapters,
        "topic": topic,
        "youtube_score": watchability_report.get("youtube_score"),
        "watchability_score": watchability_report.get("watchability_score"),
        "highlight_quality_score": watchability_report.get("highlight_quality_score"),
        "activity_profile": watchability_report.get("activity_profile"),
        "scene_threshold_used": scene_threshold_used,
        "event_count": len(events),
        "moment_count": len(moment_points),
        "whisper_cue_count": len(whisper_cues),
        "ocr_cue_count": len(ocr_cues),
    }


def _build_auto_optimize_variants(
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
            "scene_threshold": scene_threshold + 0.00,
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
        normalized = {
            "scene_threshold": max(0.08, min(0.60, float(template["scene_threshold"]))),
            "clip_before": max(3.0, min(26.0, float(template["clip_before"]))),
            "clip_after": max(5.0, min(34.0, float(template["clip_after"]))),
            "min_gap_seconds": max(6.0, min(28.0, float(template["min_gap_seconds"]))),
            "max_clips": int(max(14, min(44, int(round(float(template["max_clips"])))))),
            "target_duration_ratio": max(0.20, min(0.75, float(template["target_duration_ratio"]))),
            "ai_cue_threshold": max(0.08, min(0.80, float(template["ai_cue_threshold"]))),
            "ocr_cue_threshold": max(0.08, min(0.70, float(template["ocr_cue_threshold"]))),
        }
        signature = (
            round(float(normalized["scene_threshold"]), 4),
            round(float(normalized["clip_before"]), 3),
            round(float(normalized["clip_after"]), 3),
            round(float(normalized["min_gap_seconds"]), 3),
            float(normalized["max_clips"]),
            round(float(normalized["target_duration_ratio"]), 4),
            round(float(normalized["ai_cue_threshold"]), 4),
            round(float(normalized["ocr_cue_threshold"]), 4),
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(normalized)
        if len(deduped) >= candidate_count:
            break
    return deduped


def _select_optimize_metric_value(
    *,
    report: dict[str, object],
    metric: str,
) -> float:
    key = {
        "youtube": "youtube_score",
        "watchability": "watchability_score",
        "quality": "highlight_quality_score",
    }[metric]
    value = report.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _validate_file(input_path: Path) -> None:
    if not input_path.exists():
        raise EditorError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise EditorError(f"Input path is not a file: {input_path}")


def _require_finite(option_name: str, value: float) -> None:
    if not math.isfinite(value):
        raise EditorError(f"--{option_name} must be a finite number.")


def _validate_cli_options(args: argparse.Namespace) -> None:
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

        for option_name, option_value in (
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
            _require_finite(option_name, option_value)
            if option_value < 0:
                raise EditorError(f"--{option_name} must be >= 0.")
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
                raise EditorError("--optimize-metric must be 'youtube', 'watchability', or 'quality'.")
            if auto_optimize and optimize_candidates < 2:
                raise EditorError("--optimize-candidates must be >= 2 when --auto-optimize is enabled.")

    if args.command == "watchability":
        _require_finite("scene-threshold", args.scene_threshold)
        if not 0.0 <= args.scene_threshold <= 1.0:
            raise EditorError("--scene-threshold must be between 0.0 and 1.0.")
        for option_name, option_value in (
            ("vision-sample-fps", args.vision_sample_fps),
            ("vision-window-seconds", args.vision_window_seconds),
            ("vision-step-seconds", args.vision_step_seconds),
        ):
            _require_finite(option_name, option_value)
            if option_value <= 0:
                raise EditorError(f"--{option_name} must be > 0.")

    if args.command == "thumbnail":
        _require_finite("scene-threshold", args.scene_threshold)
        if not 0.0 <= args.scene_threshold <= 1.0:
            raise EditorError("--scene-threshold must be between 0.0 and 1.0.")
        for option_name, option_value in (
            ("vision-sample-fps", args.vision_sample_fps),
            ("vision-window-seconds", args.vision_window_seconds),
            ("vision-step-seconds", args.vision_step_seconds),
        ):
            _require_finite(option_name, option_value)
            if option_value <= 0:
                raise EditorError(f"--{option_name} must be > 0.")
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
        for option_name, option_value in (
            ("vision-sample-fps", args.vision_sample_fps),
            ("vision-window-seconds", args.vision_window_seconds),
            ("vision-step-seconds", args.vision_step_seconds),
            ("ocr-sample-fps", args.ocr_sample_fps),
            ("ocr-cue-threshold", args.ocr_cue_threshold),
            ("ai-cue-threshold", args.ai_cue_threshold),
            ("whisper-vad-threshold", args.whisper_vad_threshold),
        ):
            _require_finite(option_name, option_value)
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
        for option_name in ("crossfade-seconds", "audio-fade-seconds"):
            option_value = getattr(args, option_name.replace("-", "_"))
            _require_finite(option_name, option_value)
            if option_value < 0:
                raise EditorError(f"--{option_name} must be >= 0.")

    if args.command in {"render", "auto", "full"}:
        if not 0 <= args.crf <= 51:
            raise EditorError("--crf must be between 0 and 51.")
        if args.video_encoder not in VIDEO_ENCODERS:
            raise EditorError(f"--video-encoder must be one of: {', '.join(VIDEO_ENCODERS)}.")
        if args.video_encoder != "libx264" and args.preset != "medium":
            raise EditorError(
                "--preset applies only to libx264. Use the default preset with hardware encoders."
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lol-video-editor",
        description="Create YouTube-ready League of Legends videos from OBS recordings.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze", help="Detect highlight segments and write an edit plan JSON."
    )
    analyze.add_argument("input", type=Path, help="Path to OBS recording")
    analyze.add_argument(
        "--plan",
        type=Path,
        default=Path("edit-plan.json"),
        help="Path to write generated edit plan JSON",
    )
    analyze.add_argument("--scene-threshold", type=float, default=0.35)
    analyze.add_argument("--clip-before", type=float, default=12.0)
    analyze.add_argument("--clip-after", type=float, default=18.0)
    analyze.add_argument("--min-gap-seconds", type=float, default=20.0)
    analyze.add_argument(
        "--max-clips",
        type=int,
        default=24,
        help="Maximum number of middle highlight clips between intro and outro anchors.",
    )
    analyze.add_argument(
        "--target-duration-seconds",
        type=float,
        default=0.0,
        help="Absolute target duration override for highlights output. 0 uses --target-duration-ratio.",
    )
    analyze.add_argument(
        "--target-duration-ratio",
        type=float,
        default=DEFAULT_TARGET_DURATION_RATIO,
        help="Target highlights duration as a ratio of source match duration (default: 2/3). Set to 0 to disable ratio targeting.",
    )
    analyze.add_argument(
        "--intro-seconds",
        type=float,
        default=DEFAULT_INTRO_SECONDS,
        help="Seconds to keep after detected in-game spawn/gameplay start.",
    )
    analyze.add_argument(
        "--outro-seconds",
        type=float,
        default=DEFAULT_OUTRO_SECONDS,
        help="Seconds to keep at the end for match finish/nexus destruction.",
    )
    analyze.add_argument(
        "--end-on-result",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim analysis/rendering to the detected Victory/Defeat end screen when available.",
    )
    analyze.add_argument(
        "--result-detect-fps",
        type=float,
        default=DEFAULT_RESULT_DETECT_FPS,
        help="Frame sampling rate used to detect final Victory/Defeat end-of-match timing.",
    )
    analyze.add_argument(
        "--result-detect-tail-seconds",
        type=float,
        default=DEFAULT_RESULT_DETECT_TAIL_SECONDS,
        help="How much of the recording tail to scan for Victory/Defeat end detection.",
    )
    analyze.add_argument(
        "--one-shot-smart",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ONE_SHOT_SMART,
        help="Use one-pass adaptive tuning to keep more kill context while preserving concise pacing.",
    )
    analyze.add_argument(
        "--vision-scoring",
        type=str,
        default=DEFAULT_VISION_SCORING,
        choices=["off", "heuristic", "local-ai"],
        help="Window scoring mode. 'local-ai' combines vision scoring with local whisper.cpp transcript cues.",
    )
    analyze.add_argument(
        "--vision-sample-fps",
        type=float,
        default=DEFAULT_VISION_SAMPLE_FPS,
        help="FPS for low-resolution frame sampling used by vision scoring.",
    )
    analyze.add_argument(
        "--vision-window-seconds",
        type=float,
        default=DEFAULT_VISION_WINDOW_SECONDS,
        help="Window size used to aggregate vision activity scores.",
    )
    analyze.add_argument(
        "--vision-step-seconds",
        type=float,
        default=DEFAULT_VISION_STEP_SECONDS,
        help="Step size between consecutive vision scoring windows.",
    )
    analyze.add_argument(
        "--whisper-model",
        type=Path,
        default=None,
        help="Path to a local whisper.cpp model file (.bin/.gguf). Required for --vision-scoring local-ai.",
    )
    analyze.add_argument(
        "--whisper-bin",
        type=str,
        default=DEFAULT_WHISPER_CPP_BIN,
        help="whisper.cpp executable name/path. Use 'auto' to search PATH for whisper-cli.",
    )
    analyze.add_argument(
        "--whisper-language",
        type=str,
        default=DEFAULT_WHISPER_LANGUAGE,
        help="Language code passed to whisper.cpp (for example: en).",
    )
    analyze.add_argument(
        "--whisper-threads",
        type=int,
        default=DEFAULT_WHISPER_THREADS,
        help="Thread count for whisper.cpp inference.",
    )
    analyze.add_argument(
        "--whisper-audio-stream",
        type=int,
        default=-1,
        help=(
            "Audio stream index for whisper extraction (0-based). "
            "Use -1 for auto/default stream selection."
        ),
    )
    analyze.add_argument(
        "--whisper-vad",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_WHISPER_VAD,
        help="Enable whisper.cpp VAD segmentation for speech extraction.",
    )
    analyze.add_argument(
        "--whisper-vad-threshold",
        type=float,
        default=DEFAULT_WHISPER_VAD_THRESHOLD,
        help="VAD threshold for whisper.cpp (0-1).",
    )
    analyze.add_argument(
        "--whisper-vad-model",
        type=Path,
        default=None,
        help="Optional whisper.cpp VAD model path (required only if you want VAD enabled).",
    )
    analyze.add_argument(
        "--ocr-cue-scoring",
        type=str,
        default=DEFAULT_OCR_CUE_SCORING,
        choices=["off", "auto"],
        help="Optional OCR cue detection mode for on-screen kill/objective text.",
    )
    analyze.add_argument(
        "--tesseract-bin",
        type=str,
        default=DEFAULT_TESSERACT_BIN,
        help="tesseract executable name/path. Use 'auto' to search PATH.",
    )
    analyze.add_argument(
        "--ocr-sample-fps",
        type=float,
        default=DEFAULT_OCR_SAMPLE_FPS,
        help="Frame sampling rate for OCR cue extraction.",
    )
    analyze.add_argument(
        "--ocr-cue-threshold",
        type=float,
        default=DEFAULT_OCR_CUE_THRESHOLD,
        help="OCR cue score threshold (0-1). Lower keeps more OCR-derived cues.",
    )
    analyze.add_argument(
        "--ai-cue-threshold",
        type=float,
        default=DEFAULT_AI_CUE_THRESHOLD,
        help="Transcript cue score threshold (0-1). Lower keeps more AI-detected moments.",
    )

    render = subparsers.add_parser(
        "render", help="Render highlights MP4 from an existing edit plan JSON."
    )
    render.add_argument("input", type=Path, help="Path to OBS recording")
    render.add_argument("--plan", type=Path, required=True, help="Path to edit plan JSON")
    render.add_argument("--output", type=Path, default=Path("highlights.mp4"))
    render.add_argument("--crf", type=int, default=20)
    render.add_argument(
        "--video-encoder",
        type=str,
        default="libx264",
        choices=VIDEO_ENCODERS,
        help="Video encoder. Hardware options are available when supported by your ffmpeg build.",
    )
    render.add_argument(
        "--allow-upscale",
        action="store_true",
        help="Upscale video up to 1080p. By default, smaller sources are padded without upscaling.",
    )
    render.add_argument(
        "--auto-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-detect and crop common black borders before scaling/padding to 1080p.",
    )
    render.add_argument(
        "--crossfade-seconds",
        type=float,
        default=0.0,
        help="Crossfade duration between highlight clips. 0 disables crossfades.",
    )
    render.add_argument(
        "--audio-fade-seconds",
        type=float,
        default=0.03,
        help="Fade-in/out duration applied per clip to reduce click artifacts.",
    )
    render.add_argument(
        "--two-pass-loudnorm",
        action="store_true",
        help="Use two-pass loudness normalization (slower, more consistent output).",
    )
    render.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
        help="Encoder preset for libx264 only. Hardware encoders use their own defaults.",
    )

    auto = subparsers.add_parser(
        "auto",
        help="Analyze and render in one command (best default for first run).",
    )
    auto.add_argument("input", type=Path, help="Path to OBS recording")
    auto.add_argument("--plan", type=Path, default=Path("edit-plan.json"))
    auto.add_argument("--output", type=Path, default=Path("highlights.mp4"))
    auto.add_argument("--scene-threshold", type=float, default=0.35)
    auto.add_argument("--clip-before", type=float, default=12.0)
    auto.add_argument("--clip-after", type=float, default=18.0)
    auto.add_argument("--min-gap-seconds", type=float, default=20.0)
    auto.add_argument(
        "--max-clips",
        type=int,
        default=24,
        help="Maximum number of middle highlight clips between intro and outro anchors.",
    )
    auto.add_argument(
        "--target-duration-seconds",
        type=float,
        default=0.0,
        help="Absolute target duration override for highlights output. 0 uses --target-duration-ratio.",
    )
    auto.add_argument(
        "--target-duration-ratio",
        type=float,
        default=DEFAULT_TARGET_DURATION_RATIO,
        help="Target highlights duration as a ratio of source match duration (default: 2/3). Set to 0 to disable ratio targeting.",
    )
    auto.add_argument(
        "--intro-seconds",
        type=float,
        default=DEFAULT_INTRO_SECONDS,
        help="Seconds to keep after detected in-game spawn/gameplay start.",
    )
    auto.add_argument(
        "--outro-seconds",
        type=float,
        default=DEFAULT_OUTRO_SECONDS,
        help="Seconds to keep at the end for match finish/nexus destruction.",
    )
    auto.add_argument(
        "--end-on-result",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim analysis/rendering to the detected Victory/Defeat end screen when available.",
    )
    auto.add_argument(
        "--result-detect-fps",
        type=float,
        default=DEFAULT_RESULT_DETECT_FPS,
        help="Frame sampling rate used to detect final Victory/Defeat end-of-match timing.",
    )
    auto.add_argument(
        "--result-detect-tail-seconds",
        type=float,
        default=DEFAULT_RESULT_DETECT_TAIL_SECONDS,
        help="How much of the recording tail to scan for Victory/Defeat end detection.",
    )
    auto.add_argument(
        "--one-shot-smart",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_ONE_SHOT_SMART,
        help="Use one-pass adaptive tuning to keep more kill context while preserving concise pacing.",
    )
    auto.add_argument(
        "--vision-scoring",
        type=str,
        default=DEFAULT_VISION_SCORING,
        choices=["off", "heuristic", "local-ai"],
        help="Window scoring mode. 'local-ai' combines vision scoring with local whisper.cpp transcript cues.",
    )
    auto.add_argument(
        "--vision-sample-fps",
        type=float,
        default=DEFAULT_VISION_SAMPLE_FPS,
        help="FPS for low-resolution frame sampling used by vision scoring.",
    )
    auto.add_argument(
        "--vision-window-seconds",
        type=float,
        default=DEFAULT_VISION_WINDOW_SECONDS,
        help="Window size used to aggregate vision activity scores.",
    )
    auto.add_argument(
        "--vision-step-seconds",
        type=float,
        default=DEFAULT_VISION_STEP_SECONDS,
        help="Step size between consecutive vision scoring windows.",
    )
    auto.add_argument(
        "--whisper-model",
        type=Path,
        default=None,
        help="Path to a local whisper.cpp model file (.bin/.gguf). Required for --vision-scoring local-ai.",
    )
    auto.add_argument(
        "--whisper-bin",
        type=str,
        default=DEFAULT_WHISPER_CPP_BIN,
        help="whisper.cpp executable name/path. Use 'auto' to search PATH for whisper-cli.",
    )
    auto.add_argument(
        "--whisper-language",
        type=str,
        default=DEFAULT_WHISPER_LANGUAGE,
        help="Language code passed to whisper.cpp (for example: en).",
    )
    auto.add_argument(
        "--whisper-threads",
        type=int,
        default=DEFAULT_WHISPER_THREADS,
        help="Thread count for whisper.cpp inference.",
    )
    auto.add_argument(
        "--whisper-audio-stream",
        type=int,
        default=-1,
        help=(
            "Audio stream index for whisper extraction (0-based). "
            "Use -1 for auto/default stream selection."
        ),
    )
    auto.add_argument(
        "--whisper-vad",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_WHISPER_VAD,
        help="Enable whisper.cpp VAD segmentation for speech extraction.",
    )
    auto.add_argument(
        "--whisper-vad-threshold",
        type=float,
        default=DEFAULT_WHISPER_VAD_THRESHOLD,
        help="VAD threshold for whisper.cpp (0-1).",
    )
    auto.add_argument(
        "--whisper-vad-model",
        type=Path,
        default=None,
        help="Optional whisper.cpp VAD model path (required only if you want VAD enabled).",
    )
    auto.add_argument(
        "--ocr-cue-scoring",
        type=str,
        default=DEFAULT_OCR_CUE_SCORING,
        choices=["off", "auto"],
        help="Optional OCR cue detection mode for on-screen kill/objective text.",
    )
    auto.add_argument(
        "--tesseract-bin",
        type=str,
        default=DEFAULT_TESSERACT_BIN,
        help="tesseract executable name/path. Use 'auto' to search PATH.",
    )
    auto.add_argument(
        "--ocr-sample-fps",
        type=float,
        default=DEFAULT_OCR_SAMPLE_FPS,
        help="Frame sampling rate for OCR cue extraction.",
    )
    auto.add_argument(
        "--ocr-cue-threshold",
        type=float,
        default=DEFAULT_OCR_CUE_THRESHOLD,
        help="OCR cue score threshold (0-1). Lower keeps more OCR-derived cues.",
    )
    auto.add_argument(
        "--ai-cue-threshold",
        type=float,
        default=DEFAULT_AI_CUE_THRESHOLD,
        help="Transcript cue score threshold (0-1). Lower keeps more AI-detected moments.",
    )
    auto.add_argument("--crf", type=int, default=20)
    auto.add_argument(
        "--video-encoder",
        type=str,
        default="libx264",
        choices=VIDEO_ENCODERS,
        help="Video encoder. Hardware options are available when supported by your ffmpeg build.",
    )
    auto.add_argument(
        "--allow-upscale",
        action="store_true",
        help="Upscale video up to 1080p. By default, smaller sources are padded without upscaling.",
    )
    auto.add_argument(
        "--auto-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-detect and crop common black borders before scaling/padding to 1080p.",
    )
    auto.add_argument(
        "--crossfade-seconds",
        type=float,
        default=0.0,
        help="Crossfade duration between highlight clips. 0 disables crossfades.",
    )
    auto.add_argument(
        "--audio-fade-seconds",
        type=float,
        default=0.03,
        help="Fade-in/out duration applied per clip to reduce click artifacts.",
    )
    auto.add_argument(
        "--two-pass-loudnorm",
        action="store_true",
        help="Use two-pass loudness normalization (slower, more consistent output).",
    )
    auto.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
        help="Encoder preset for libx264 only. Hardware encoders use their own defaults.",
    )
    auto.add_argument(
        "--auto-optimize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Try multiple parameter variants and choose the best by --optimize-metric.",
    )
    auto.add_argument(
        "--optimize-candidates",
        type=int,
        default=DEFAULT_AUTO_OPTIMIZE_CANDIDATES,
        help="Number of auto-optimize candidates to evaluate when --auto-optimize is enabled.",
    )
    auto.add_argument(
        "--optimize-metric",
        type=str,
        default=DEFAULT_AUTO_OPTIMIZE_METRIC,
        choices=["youtube", "watchability", "quality"],
        help="Metric used by auto-optimize candidate selection.",
    )

    full = subparsers.add_parser(
        "full",
        help="Transcode full match to YouTube-ready format without clipping highlights.",
    )
    full.add_argument("input", type=Path, help="Path to OBS recording")
    full.add_argument("--output", type=Path, default=Path("full-match-youtube.mp4"))
    full.add_argument("--crf", type=int, default=20)
    full.add_argument(
        "--video-encoder",
        type=str,
        default="libx264",
        choices=VIDEO_ENCODERS,
        help="Video encoder. Hardware options are available when supported by your ffmpeg build.",
    )
    full.add_argument(
        "--allow-upscale",
        action="store_true",
        help="Upscale video up to 1080p. By default, smaller sources are padded without upscaling.",
    )
    full.add_argument(
        "--auto-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-detect and crop common black borders before scaling/padding to 1080p.",
    )
    full.add_argument(
        "--two-pass-loudnorm",
        action="store_true",
        help="Use two-pass loudness normalization (slower, more consistent output).",
    )
    full.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
        help="Encoder preset for libx264 only. Hardware encoders use their own defaults.",
    )

    thumbnail = subparsers.add_parser(
        "thumbnail",
        help="Generate a YouTube thumbnail frame from a video.",
    )
    thumbnail.add_argument("input", type=Path, help="Path to source video")
    thumbnail.add_argument("--output", type=Path, default=Path("thumbnail.jpg"))
    thumbnail.add_argument(
        "--timestamp",
        type=float,
        default=None,
        help="Optional timestamp in seconds. If omitted, auto-selects a high-action frame.",
    )
    thumbnail.add_argument(
        "--scene-threshold",
        type=float,
        default=DEFAULT_THUMBNAIL_SCENE_THRESHOLD,
        help="Scene threshold used while auto-selecting thumbnail frames.",
    )
    thumbnail.add_argument(
        "--vision-sample-fps",
        type=float,
        default=DEFAULT_THUMBNAIL_VISION_SAMPLE_FPS,
        help="Frame sampling FPS used while auto-selecting thumbnail frames.",
    )
    thumbnail.add_argument(
        "--vision-window-seconds",
        type=float,
        default=DEFAULT_THUMBNAIL_VISION_WINDOW_SECONDS,
        help="Window duration for auto thumbnail activity scoring.",
    )
    thumbnail.add_argument(
        "--vision-step-seconds",
        type=float,
        default=DEFAULT_THUMBNAIL_VISION_STEP_SECONDS,
        help="Step between windows for auto thumbnail activity scoring.",
    )
    thumbnail.add_argument(
        "--width",
        type=int,
        default=DEFAULT_THUMBNAIL_WIDTH,
        help="Thumbnail output width in pixels.",
    )
    thumbnail.add_argument(
        "--height",
        type=int,
        default=DEFAULT_THUMBNAIL_HEIGHT,
        help="Thumbnail output height in pixels.",
    )
    thumbnail.add_argument(
        "--quality",
        type=int,
        default=DEFAULT_THUMBNAIL_QUALITY,
        help="JPEG quality (2 best, 31 worst). Ignored for PNG output.",
    )
    thumbnail.add_argument(
        "--champion-overlay",
        type=Path,
        default=None,
        help="Path to transparent champion PNG that should be overlaid on the thumbnail.",
    )
    thumbnail.add_argument(
        "--champion-scale",
        type=float,
        default=DEFAULT_THUMBNAIL_CHAMPION_SCALE,
        help="Champion overlay width as a ratio of thumbnail width.",
    )
    thumbnail.add_argument(
        "--champion-anchor",
        type=str,
        default=DEFAULT_THUMBNAIL_CHAMPION_ANCHOR,
        choices=["left", "center", "right"],
        help="Horizontal anchor for champion overlay placement.",
    )
    thumbnail.add_argument(
        "--headline",
        type=str,
        default=None,
        help="Optional text to overlay on top of the thumbnail. Supports \\n for line breaks.",
    )
    thumbnail.add_argument(
        "--headline-size",
        type=int,
        default=DEFAULT_THUMBNAIL_HEADLINE_SIZE,
        help="Headline font size in pixels.",
    )
    thumbnail.add_argument(
        "--headline-color",
        type=str,
        default=DEFAULT_THUMBNAIL_HEADLINE_COLOR,
        help="Headline font color (for example: white, yellow, #ffd400).",
    )
    thumbnail.add_argument(
        "--headline-font",
        type=Path,
        default=None,
        help="Optional path to a .ttf/.otf font file used for headline text.",
    )
    thumbnail.add_argument(
        "--headline-y-ratio",
        type=float,
        default=DEFAULT_THUMBNAIL_HEADLINE_Y_RATIO,
        help="Vertical headline position as a ratio of output height.",
    )
    thumbnail.add_argument(
        "--auto-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-detect and crop common black borders before resizing.",
    )
    thumbnail.add_argument(
        "--enhance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply lightweight contrast/saturation/sharpen enhancement.",
    )

    description = subparsers.add_parser(
        "description",
        help="Generate CTR-focused YouTube title/description text from video events and cues.",
    )
    description.add_argument("input", type=Path, help="Path to source video")
    description.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_DESCRIPTION_OUTPUT),
        help="Path to write generated title ideas + description text.",
    )
    description.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write a JSON metadata report for generated description assets.",
    )
    description.add_argument(
        "--champion",
        type=str,
        default="",
        help="Champion/player focus to bias title and hashtags (for example: Zed).",
    )
    description.add_argument(
        "--channel-name",
        type=str,
        default="",
        help="Optional channel name for subscribe CTA line.",
    )
    description.add_argument("--scene-threshold", type=float, default=0.20)
    description.add_argument(
        "--vision-sample-fps",
        type=float,
        default=DEFAULT_VISION_SAMPLE_FPS,
        help="FPS for low-resolution frame sampling used by scoring.",
    )
    description.add_argument(
        "--vision-window-seconds",
        type=float,
        default=DEFAULT_VISION_WINDOW_SECONDS,
        help="Window size used to aggregate gameplay activity scores.",
    )
    description.add_argument(
        "--vision-step-seconds",
        type=float,
        default=DEFAULT_VISION_STEP_SECONDS,
        help="Step size between consecutive scoring windows.",
    )
    description.add_argument(
        "--whisper-model",
        type=Path,
        default=None,
        help="Optional path to local whisper.cpp model for transcript cue extraction.",
    )
    description.add_argument(
        "--whisper-bin",
        type=str,
        default=DEFAULT_WHISPER_CPP_BIN,
        help="whisper.cpp executable name/path. Use 'auto' to search PATH for whisper-cli.",
    )
    description.add_argument(
        "--whisper-language",
        type=str,
        default=DEFAULT_WHISPER_LANGUAGE,
        help="Language code passed to whisper.cpp (for example: en).",
    )
    description.add_argument(
        "--whisper-threads",
        type=int,
        default=DEFAULT_WHISPER_THREADS,
        help="Thread count for whisper.cpp inference.",
    )
    description.add_argument(
        "--whisper-audio-stream",
        type=int,
        default=-1,
        help="Audio stream index for whisper extraction (0-based). Use -1 for auto/default.",
    )
    description.add_argument(
        "--whisper-vad",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_WHISPER_VAD,
        help="Enable whisper.cpp VAD segmentation for speech extraction.",
    )
    description.add_argument(
        "--whisper-vad-threshold",
        type=float,
        default=DEFAULT_WHISPER_VAD_THRESHOLD,
        help="VAD threshold for whisper.cpp (0-1).",
    )
    description.add_argument(
        "--whisper-vad-model",
        type=Path,
        default=None,
        help="Optional whisper.cpp VAD model path (used when --whisper-vad is enabled).",
    )
    description.add_argument(
        "--ocr-cue-scoring",
        type=str,
        default=DEFAULT_OCR_CUE_SCORING,
        choices=["off", "auto"],
        help="Enable OCR cue extraction for on-screen objective/kill text.",
    )
    description.add_argument(
        "--tesseract-bin",
        type=str,
        default=DEFAULT_TESSERACT_BIN,
        help="tesseract executable name/path. Use 'auto' to search PATH.",
    )
    description.add_argument(
        "--ocr-sample-fps",
        type=float,
        default=DEFAULT_OCR_SAMPLE_FPS,
        help="Frame sampling rate for OCR cue extraction.",
    )
    description.add_argument(
        "--ocr-cue-threshold",
        type=float,
        default=DEFAULT_OCR_CUE_THRESHOLD,
        help="OCR cue score threshold (0-1). Lower keeps more OCR cues.",
    )
    description.add_argument(
        "--ai-cue-threshold",
        type=float,
        default=DEFAULT_AI_CUE_THRESHOLD,
        help="Transcript cue score threshold (0-1). Lower keeps more AI transcript cues.",
    )
    description.add_argument(
        "--max-moments",
        type=int,
        default=DEFAULT_DESCRIPTION_MAX_MOMENTS,
        help="Maximum number of timestamped moments to include in the generated description.",
    )
    description.add_argument(
        "--title-count",
        type=int,
        default=DEFAULT_DESCRIPTION_TITLE_COUNT,
        help="Number of title ideas to generate.",
    )

    watchability = subparsers.add_parser(
        "watchability",
        help="Analyze a video and report a heuristic watchability score.",
    )
    watchability.add_argument("input", type=Path, help="Path to rendered video")
    watchability.add_argument("--scene-threshold", type=float, default=0.35)
    watchability.add_argument(
        "--vision-sample-fps",
        type=float,
        default=DEFAULT_VISION_SAMPLE_FPS,
        help="FPS for low-resolution frame sampling used by watchability analysis.",
    )
    watchability.add_argument(
        "--vision-window-seconds",
        type=float,
        default=DEFAULT_VISION_WINDOW_SECONDS,
        help="Window size used to aggregate watchability activity scores.",
    )
    watchability.add_argument(
        "--vision-step-seconds",
        type=float,
        default=DEFAULT_VISION_STEP_SECONDS,
        help="Step size between consecutive watchability windows.",
    )
    watchability.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write a JSON watchability report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        _validate_cli_options(args)
        _require_binary("ffmpeg")
        _require_binary("ffprobe")
        _validate_file(args.input)

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

        if args.command == "auto":
            def run_auto_analysis(
                *,
                plan_path: Path,
                overrides: dict[str, float | int] | None = None,
            ) -> dict[str, int | float | bool]:
                options = overrides or {}
                return analyze_recording(
                    input_path=args.input,
                    plan_path=plan_path,
                    scene_threshold=float(options.get("scene_threshold", args.scene_threshold)),
                    clip_before=float(options.get("clip_before", args.clip_before)),
                    clip_after=float(options.get("clip_after", args.clip_after)),
                    min_gap_seconds=float(options.get("min_gap_seconds", args.min_gap_seconds)),
                    max_clips=int(options.get("max_clips", args.max_clips)),
                    target_duration_seconds=args.target_duration_seconds,
                    target_duration_ratio=float(
                        options.get("target_duration_ratio", args.target_duration_ratio)
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
                    ocr_cue_threshold=float(options.get("ocr_cue_threshold", args.ocr_cue_threshold)),
                    ai_cue_threshold=float(options.get("ai_cue_threshold", args.ai_cue_threshold)),
                    end_on_result=args.end_on_result,
                    result_detect_fps=args.result_detect_fps,
                    result_detect_tail_seconds=args.result_detect_tail_seconds,
                    one_shot_smart=args.one_shot_smart,
                )

            selected_stats: dict[str, int | float | bool]
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
                    (
                        f"Auto-optimize enabled: evaluating {len(variants)} candidates "
                        f"by {args.optimize_metric} score."
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                best_candidate: dict[str, object] | None = None
                with tempfile.TemporaryDirectory(prefix="lol-optimize-") as temp_dir:
                    temp_dir_path = Path(temp_dir)
                    for index, variant in enumerate(variants, start=1):
                        candidate_plan = temp_dir_path / f"candidate-{index}.json"
                        candidate_output = temp_dir_path / f"candidate-{index}.mp4"
                        print(
                            f"Auto-optimize candidate {index}/{len(variants)}...",
                            file=sys.stderr,
                            flush=True,
                        )
                        candidate_stats = run_auto_analysis(
                            plan_path=candidate_plan,
                            overrides=variant,
                        )
                        print("Rendering highlights...", file=sys.stderr, flush=True)
                        render_highlights(
                            input_path=args.input,
                            plan_path=candidate_plan,
                            output_path=candidate_output,
                            crf=args.crf,
                            preset=args.preset,
                            video_encoder=args.video_encoder,
                            allow_upscale=args.allow_upscale,
                            auto_crop=args.auto_crop,
                            crossfade_seconds=args.crossfade_seconds,
                            audio_fade_seconds=args.audio_fade_seconds,
                            two_pass_loudnorm=args.two_pass_loudnorm,
                        )
                        candidate_report = analyze_watchability(
                            input_path=candidate_output,
                            scene_threshold=min(0.20, args.scene_threshold),
                            vision_sample_fps=args.vision_sample_fps,
                            vision_window_seconds=args.vision_window_seconds,
                            vision_step_seconds=args.vision_step_seconds,
                            show_progress=False,
                        )
                        metric_value = _select_optimize_metric_value(
                            report=candidate_report,
                            metric=args.optimize_metric,
                        )
                        print(
                            (
                                f"  candidate {index} -> "
                                f"youtube={candidate_report['youtube_score']}/100 | "
                                f"watchability={candidate_report['watchability_score']}/100 | "
                                f"quality={candidate_report['highlight_quality_score']}/100"
                            ),
                            file=sys.stderr,
                            flush=True,
                        )
                        if best_candidate is None or metric_value > float(best_candidate["metric_value"]):
                            best_candidate = {
                                "metric_value": metric_value,
                                "plan_path": candidate_plan,
                                "stats": candidate_stats,
                                "report": candidate_report,
                                "variant": variant,
                            }
                    if best_candidate is None:
                        raise EditorError("Auto-optimize failed to evaluate candidates.")
                    shutil.copy2(Path(best_candidate["plan_path"]), args.plan)
                    selected_stats = dict(best_candidate["stats"])  # type: ignore[arg-type]
                    selected_report = dict(best_candidate["report"])  # type: ignore[arg-type]
                    print(
                        (
                            "Auto-optimize selected best candidate: "
                            f"{args.optimize_metric}={float(best_candidate['metric_value']):.2f}."
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
            else:
                selected_stats = run_auto_analysis(plan_path=args.plan)
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
                f"[transcript={selected_stats['whisper_cue_count']}, ocr={selected_stats['ocr_cue_count']}], "
                f"plan: {args.plan})."
            )
            if selected_report is not None:
                print(
                    (
                        f"Auto-optimize final estimate: youtube={selected_report['youtube_score']}/100 | "
                        f"watchability={selected_report['watchability_score']}/100 | "
                        f"quality={selected_report['highlight_quality_score']}/100"
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            return 0

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

        if args.command == "watchability":
            report = analyze_watchability(
                input_path=args.input,
                scene_threshold=args.scene_threshold,
                vision_sample_fps=args.vision_sample_fps,
                vision_window_seconds=args.vision_window_seconds,
                vision_step_seconds=args.vision_step_seconds,
            )
            print(
                f"Watchability score: {report['watchability_score']}/100 "
                f"({report['rating']})."
            )
            print(
                f"Highlight quality: {report['highlight_quality_score']}/100 "
                f"({report['quality_rating']})."
            )
            blend = report.get("score_blend", {})
            if isinstance(blend, dict):
                watch_w = blend.get("watchability_weight")
                quality_w = blend.get("quality_weight")
                print(
                    f"YouTube score: {report['youtube_score']}/100 "
                    f"(watchability={watch_w}, quality={quality_w})."
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
                for recommendation in recommendations:
                    if isinstance(recommendation, str):
                        print(f"- {recommendation}")
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
