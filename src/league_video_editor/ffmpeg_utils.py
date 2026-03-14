"""Low-level ffmpeg / ffprobe helpers.

All subprocess calls to ffmpeg/ffprobe live here.  Higher-level modules (analyze,
render, thumbnail) import from this module.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
from typing import Callable

from .models import (
    CROPDETECT_PATTERN,
    FFMPEG_TIME_PATTERN,
    LOUDNORM_ANALYSIS_FILTER,
    LOUDNORM_FILTER,
    LOUDNORM_NONFINITE_PATTERN,
    SCENE_PTS_PATTERN,
    SIGNALSTATS_FRAME_PATTERN,
    SIGNALSTATS_SATAVG_PATTERN,
    SIGNALSTATS_YDIF_PATTERN,
    VIDEO_ENCODERS,
    EditorError,
    Segment,
    VisionWindow,
)


# ---------------------------------------------------------------------------
# Binary checks
# ---------------------------------------------------------------------------


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise EditorError(
            f"'{name}' is required but not installed. Install ffmpeg/ffprobe first."
        )


def resolve_whisper_cpp_binary(configured_binary: str) -> str | None:
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


def resolve_tesseract_binary(configured_binary: str) -> str | None:
    binary_value = configured_binary.strip()
    if binary_value and binary_value != "auto":
        resolved = shutil.which(binary_value)
        if resolved is not None:
            return resolved
        if Path(binary_value).exists():
            return binary_value
        return None
    return shutil.which("tesseract")


# ---------------------------------------------------------------------------
# Encoder auto-detection
# ---------------------------------------------------------------------------


def detect_available_video_encoders() -> list[str]:
    """Return video encoder names supported by the local ffmpeg build."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout + result.stderr
        return [enc for enc in VIDEO_ENCODERS if enc in output]
    except (subprocess.SubprocessError, OSError):
        return ["libx264"]


_cached_encoders: list[str] | None = None


def get_available_encoders() -> list[str]:
    """Cached version of :func:`detect_available_video_encoders`."""
    global _cached_encoders
    if _cached_encoders is None:
        _cached_encoders = detect_available_video_encoders()
    return _cached_encoders


def resolve_encoder(requested: str) -> str:
    """Resolve ``"auto"`` to the best encoder available.

    Priority on Apple Silicon:  h264_videotoolbox → libx264
    Priority elsewhere:         libx264
    """
    if requested != "auto":
        return requested
    available = get_available_encoders()
    for preferred in ("h264_videotoolbox", "libx264"):
        if preferred in available:
            return preferred
    # Final fallback – libx264 is always required for this tool
    return "libx264"


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def run_command(
    cmd: list[str], *, capture_output: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=capture_output,
    )


def render_progress_line(label: str, ratio: float, *, width: int = 28) -> str:
    clamped = min(1.0, max(0.0, ratio))
    filled = int(clamped * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"\r{label}: [{bar}] {clamped * 100:5.1f}%"


def parse_ffmpeg_clock_to_seconds(clock: str) -> float:
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


# ---------------------------------------------------------------------------
# Video / audio probing
# ---------------------------------------------------------------------------


def probe_duration_seconds(input_path: Path, *, _run_cmd_fn=None) -> float:
    _do_run = _run_cmd_fn if _run_cmd_fn is not None else run_command
    result = _do_run(
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
        raise EditorError(
            f"Could not parse duration from {input_path}: {duration_text!r}"
        ) from error
    if not math.isfinite(duration) or duration < 0:
        raise EditorError(
            f"Duration is not a valid finite value for {input_path}: {duration_text!r}"
        )
    return duration


def has_audio_stream(input_path: Path) -> bool:
    result = run_command(
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


# ---------------------------------------------------------------------------
# Scene event detection
# ---------------------------------------------------------------------------


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
                current_seconds = parse_ffmpeg_clock_to_seconds(time_match.group(1))
                progress_ratio = min(1.0, current_seconds / duration_seconds)
                if progress_ratio >= last_output_ratio + 0.01 or progress_ratio >= 1.0:
                    if progress_label is not None:
                        print(
                            render_progress_line(progress_label, progress_ratio),
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
                    render_progress_line(progress_label, 1.0),
                    file=sys.stderr,
                    flush=True,
                )
            if progress_callback is not None:
                progress_callback(1.0)
        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code, cmd, stderr="".join(stderr_tail)
            )
        return sorted(events)
    except KeyboardInterrupt:
        process.terminate()
        raise


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
            duration_seconds = probe_duration_seconds(input_path)
        return _detect_scene_events_with_progress(
            cmd,
            duration_seconds=duration_seconds,
            progress_label=progress_label,
            progress_callback=progress_callback,
        )
    result = run_command(cmd, capture_output=True)
    output = (result.stderr or "") + "\n" + (result.stdout or "")
    matches = SCENE_PTS_PATTERN.findall(output)
    return sorted({float(v) for v in matches if float(v) >= 0.0})


def detect_scene_events_adaptive(
    input_path: Path,
    scene_threshold: float,
    *,
    duration_seconds: float | None = None,
    show_progress: bool = False,
    progress_label: str | None = "Analyzing scenes",
    progress_callback: Callable[[float], None] | None = None,
    minimum_threshold: float = 0.08,
    factors: tuple[float, ...] = (1.0, 0.75, 0.6, 0.45),
    _detect_fn=None,
) -> tuple[list[float], float]:
    _do_detect = _detect_fn if _detect_fn is not None else detect_scene_events
    clamped = min(1.0, max(0.0, scene_threshold))
    thresholds: list[float] = []
    for factor in factors:
        candidate = max(minimum_threshold, clamped * factor)
        if not any(abs(candidate - t) < 1e-6 for t in thresholds):
            thresholds.append(candidate)
    if not thresholds:
        thresholds.append(clamped)

    for index, threshold in enumerate(thresholds):
        attempt_label = progress_label
        if show_progress and progress_label is not None and index > 0:
            attempt_label = f"{progress_label} retry {index + 1}"
        events = _do_detect(
            input_path,
            threshold,
            duration_seconds=duration_seconds,
            show_progress=show_progress,
            progress_label=attempt_label,
            progress_callback=progress_callback,
        )
        if events:
            if index > 0:
                if show_progress and progress_label is not None:
                    print(file=sys.stderr)
                print(
                    f"Scene threshold adjusted to {threshold:.3f}; "
                    f"detected {len(events)} scene events.",
                    file=sys.stderr,
                    flush=True,
                )
            return events, threshold
        if index + 1 < len(thresholds):
            next_threshold = thresholds[index + 1]
            if show_progress and progress_label is not None:
                print(file=sys.stderr)
            print(
                f"No scene events at threshold {threshold:.3f}; "
                f"retrying with {next_threshold:.3f}.",
                file=sys.stderr,
                flush=True,
            )

    return [], thresholds[-1] if thresholds else clamped


# ---------------------------------------------------------------------------
# Cropdetect
# ---------------------------------------------------------------------------


def detect_crop_filter(
    input_path: Path,
    *,
    duration_seconds: float,
    _run_cmd_fn=None,
) -> str | None:
    _do_run = _run_cmd_fn if _run_cmd_fn is not None else run_command
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
        result = _do_run(cmd, capture_output=True)
    except subprocess.CalledProcessError:
        return None
    output = (result.stderr or "") + "\n" + (result.stdout or "")
    matches = CROPDETECT_PATTERN.findall(output)
    if not matches:
        return None
    crop_counts: dict[tuple[int, int, int, int], int] = {}
    for w, h, x, y in matches:
        width, height, x_off, y_off = int(w), int(h), int(x), int(y)
        if width <= 0 or height <= 0 or x_off < 0 or y_off < 0:
            continue
        key = (width, height, x_off, y_off)
        crop_counts[key] = crop_counts.get(key, 0) + 1
    if not crop_counts:
        return None
    best, _ = max(crop_counts.items(), key=lambda item: (item[1], item[0][0] * item[0][1]))
    w, h, x, y = best
    return f"crop={w}:{h}:{x}:{y}"


# ---------------------------------------------------------------------------
# Signalstats (motion + saturation)
# ---------------------------------------------------------------------------


def extract_signalstats_samples(
    input_path: Path,
    *,
    duration_seconds: float,
    sample_fps: float,
    crop_filter: str | None = None,
    show_progress: bool = False,
    progress_label: str | None = "Scoring gameplay",
    progress_callback: Callable[[float], None] | None = None,
) -> list[tuple[float, float, float]]:
    if show_progress and progress_callback is not None:
        progress_callback(0.01)

    signal_filters: list[str] = []
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

            sat_match = SIGNALSTATS_SATAVG_PATTERN.search(line)
            if sat_match:
                try:
                    current_saturation = float(sat_match.group(1))
                except ValueError:
                    current_saturation = None

            if show_progress and duration_seconds > 0:
                progress_time = current_time
                time_match = FFMPEG_TIME_PATTERN.search(line)
                if progress_time is None and time_match:
                    progress_time = parse_ffmpeg_clock_to_seconds(time_match.group(1))
                if progress_time is not None:
                    ratio = min(1.0, max(0.0, progress_time / duration_seconds))
                    if ratio >= last_output_ratio + 0.01 or ratio >= 1.0:
                        if progress_label is not None:
                            print(
                                render_progress_line(progress_label, ratio),
                                end="",
                                file=sys.stderr,
                                flush=True,
                            )
                        if progress_callback is not None:
                            progress_callback(ratio)
                        last_output_ratio = ratio

        if current_time is not None and current_motion is not None and current_saturation is not None:
            samples.append((current_time, current_motion, current_saturation))

        return_code = process.wait()
        if show_progress and duration_seconds > 0:
            if progress_label is not None:
                print(
                    render_progress_line(progress_label, 1.0),
                    file=sys.stderr,
                    flush=True,
                )
            if progress_callback is not None:
                progress_callback(1.0)
        if return_code != 0:
            raise subprocess.CalledProcessError(
                return_code, cmd, stderr="".join(stderr_tail)
            )
        return samples
    except KeyboardInterrupt:
        process.terminate()
        raise


# ---------------------------------------------------------------------------
# Audio extraction (for whisper)
# ---------------------------------------------------------------------------


def extract_audio_for_whisper(
    *,
    input_path: Path,
    output_audio_path: Path,
    audio_stream_index: int | None = None,
    _run_cmd_fn=None,
) -> None:
    _do_run = _run_cmd_fn if _run_cmd_fn is not None else run_command
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
    _do_run(cmd, capture_output=True)


# ---------------------------------------------------------------------------
# OCR frame extraction
# ---------------------------------------------------------------------------


def extract_ocr_frames(
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
    run_command(cmd, capture_output=True)
    return sorted(output_dir.glob("frame_*.jpg"))


def extract_result_ocr_frames(
    *,
    input_path: Path,
    output_dir: Path,
    sample_fps: float,
    start_time: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "frame_%06d.jpg"
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
    run_command(cmd, capture_output=True)
    return sorted(output_dir.glob("frame_*.jpg"))


# ---------------------------------------------------------------------------
# Tesseract
# ---------------------------------------------------------------------------


def run_tesseract_ocr(*, image_path: Path, tesseract_binary: str) -> str:
    cmd = [
        tesseract_binary,
        str(image_path),
        "stdout",
        "--psm",
        "6",
        "-l",
        "eng",
    ]
    result = run_command(cmd, capture_output=True)
    return (result.stdout or "").strip()


# ---------------------------------------------------------------------------
# Render filter graph helpers
# ---------------------------------------------------------------------------


def video_postprocess_filter(
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


def video_codec_args(*, video_encoder: str, crf: int, preset: str) -> list[str]:
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


def effective_transition_duration(
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


def build_filter_complex(
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
    durations = [s.duration for s in segments]

    video_output_label: str | None = None
    if include_video:
        for index, segment in enumerate(segments):
            pieces.append(
                f"[0:v]trim=start={segment.start:.3f}:end={segment.end:.3f},"
                f"setpts=PTS-STARTPTS[v{index}]"
            )
        if crossfade_seconds > 0 and len(segments) > 1:
            running_duration = durations[0]
            current_label = "v0"
            for index in range(1, len(segments)):
                td = effective_transition_duration(
                    crossfade_seconds,
                    left_duration=running_duration,
                    right_duration=durations[index],
                )
                offset = max(0.0, running_duration - td)
                next_label = f"vxf{index}"
                pieces.append(
                    f"[{current_label}][v{index}]"
                    f"xfade=transition=fade:duration={td:.3f}:offset={offset:.3f}"
                    f"[{next_label}]"
                )
                current_label = next_label
                running_duration = running_duration + durations[index] - td
            video_source = current_label
        else:
            concat_inputs = "".join(f"[v{i}]" for i in range(len(segments)))
            pieces.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=0[vcat]")
            video_source = "vcat"
        pieces.append(
            f"[{video_source}]"
            f"{video_postprocess_filter(allow_upscale=allow_upscale, crop_filter=crop_filter)}"
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
                fade_d = min(audio_fade_seconds, segment.duration / 2)
                if fade_d > 0:
                    fade_out_start = max(0.0, segment.duration - fade_d)
                    audio_steps.extend(
                        [
                            f"afade=t=in:st=0:d={fade_d:.3f}",
                            f"afade=t=out:st={fade_out_start:.3f}:d={fade_d:.3f}",
                        ]
                    )
            pieces.append(f"[0:a]{','.join(audio_steps)}[a{index}]")

        if crossfade_seconds > 0 and len(segments) > 1:
            running_duration = durations[0]
            current_label = "a0"
            for index in range(1, len(segments)):
                td = effective_transition_duration(
                    crossfade_seconds,
                    left_duration=running_duration,
                    right_duration=durations[index],
                )
                next_label = f"axf{index}"
                pieces.append(
                    f"[{current_label}][a{index}]"
                    f"acrossfade=d={td:.3f}:c1=tri:c2=tri"
                    f"[{next_label}]"
                )
                current_label = next_label
                running_duration = running_duration + durations[index] - td
            audio_source = current_label
        else:
            concat_inputs = "".join(f"[a{i}]" for i in range(len(segments)))
            pieces.append(f"{concat_inputs}concat=n={len(segments)}:v=0:a=1[acat]")
            audio_source = "acat"

        post_filter = audio_post_filter or "anull"
        pieces.append(f"[{audio_source}]{post_filter}[aout]")
        audio_output_label = "[aout]"

    return ";".join(pieces), video_output_label, audio_output_label


def estimate_render_duration(segments: list[Segment], *, crossfade_seconds: float) -> float:
    if not segments:
        return 0.0
    durations = [s.duration for s in segments]
    if crossfade_seconds <= 0 or len(durations) == 1:
        return max(0.0, sum(durations))
    running = durations[0]
    for index in range(1, len(durations)):
        td = effective_transition_duration(
            crossfade_seconds,
            left_duration=running,
            right_duration=durations[index],
        )
        running = running + durations[index] - td
    return max(0.0, running)


# ---------------------------------------------------------------------------
# Loudnorm helpers
# ---------------------------------------------------------------------------


def looks_like_loudnorm_nonfinite_error(error: subprocess.CalledProcessError) -> bool:
    combined = (error.stderr or "") + "\n" + (error.stdout or "")
    return bool(LOUDNORM_NONFINITE_PATTERN.search(combined))


def extract_loudnorm_json(analysis_output: str) -> dict[str, float]:
    required = {
        "input_i": "measured_I",
        "input_tp": "measured_TP",
        "input_lra": "measured_LRA",
        "input_thresh": "measured_thresh",
        "target_offset": "offset",
    }
    # Extract JSON objects from output
    json_candidates: list[str] = []
    depth = 0
    object_start: int | None = None
    for idx, ch in enumerate(analysis_output):
        if ch == "{":
            if depth == 0:
                object_start = idx
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and object_start is not None:
                json_candidates.append(analysis_output[object_start : idx + 1])
                object_start = None

    if not json_candidates:
        raise EditorError("Could not parse loudnorm analysis output.")

    payload: dict[str, object] | None = None
    for candidate in reversed(json_candidates):
        try:
            decoded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict) and all(k in decoded for k in required):
            payload = decoded
            break

    if payload is None:
        raise EditorError("Could not decode loudnorm analysis JSON.")

    extracted: dict[str, float] = {}
    for source_key, output_key in required.items():
        try:
            value = float(payload[source_key])  # type: ignore[arg-type]
        except (TypeError, ValueError) as err:
            raise EditorError(f"Invalid '{source_key}' in loudnorm analysis.") from err
        if not math.isfinite(value):
            raise EditorError(f"Non-finite '{source_key}' in loudnorm analysis.")
        extracted[output_key] = value
    return extracted


def build_two_pass_loudnorm_filter(analysis_command: list[str], *, _run_cmd_fn=None) -> str:
    _do_run = _run_cmd_fn if _run_cmd_fn is not None else run_command
    result = _do_run(analysis_command, capture_output=True)
    analysis_output = (result.stderr or "") + "\n" + (result.stdout or "")
    metrics = extract_loudnorm_json(analysis_output)
    return (
        "loudnorm=I=-14:LRA=11:TP=-1.5:"
        f"measured_I={metrics['measured_I']:.3f}:"
        f"measured_TP={metrics['measured_TP']:.3f}:"
        f"measured_LRA={metrics['measured_LRA']:.3f}:"
        f"measured_thresh={metrics['measured_thresh']:.3f}:"
        f"offset={metrics['offset']:.3f}:"
        "linear=true:print_format=summary"
    )


def select_loudnorm_filter(
    *,
    two_pass_loudnorm: bool,
    analysis_command: list[str] | None,
    _run_cmd_fn=None,
) -> str:
    if not two_pass_loudnorm or analysis_command is None:
        return LOUDNORM_FILTER
    try:
        return build_two_pass_loudnorm_filter(analysis_command, _run_cmd_fn=_run_cmd_fn)
    except (EditorError, subprocess.CalledProcessError) as error:
        print(
            f"Warning: two-pass loudnorm analysis failed ({error}); using one-pass loudnorm.",
            file=sys.stderr,
        )
        return LOUDNORM_FILTER


def run_with_loudnorm_fallback(
    command_with_loudnorm: list[str],
    command_without_loudnorm: list[str] | None,
    *,
    progress_label: str | None = None,
    progress_duration_seconds: float | None = None,
    _run_stderr_fn=None,
) -> None:
    _do_run = _run_stderr_fn if _run_stderr_fn is not None else _run_command_with_stderr_tail
    if command_without_loudnorm is None:
        _do_run(
            command_with_loudnorm,
            progress_label=progress_label,
            duration_seconds=progress_duration_seconds,
        )
        return
    try:
        _do_run(
            command_with_loudnorm,
            progress_label=progress_label,
            duration_seconds=progress_duration_seconds,
        )
    except subprocess.CalledProcessError as error:
        if not looks_like_loudnorm_nonfinite_error(error):
            raise
        print(
            "Warning: loudnorm failed (non-finite audio); retrying without loudness normalization.",
            file=sys.stderr,
        )
        _do_run(
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
                current_seconds = parse_ffmpeg_clock_to_seconds(time_match.group(1))
                ratio = min(1.0, max(0.0, current_seconds / duration_seconds))
                if ratio >= last_output_ratio + 0.01 or ratio >= 1.0:
                    print(
                        render_progress_line(progress_label, ratio),
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_output_ratio = ratio
            return_code = process.wait()
            print(
                render_progress_line(progress_label, 1.0),
                file=sys.stderr,
                flush=True,
            )
            if return_code != 0:
                raise subprocess.CalledProcessError(
                    return_code, cmd, stderr="".join(stderr_tail)
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
        tail = stderr_file.read()
        raise subprocess.CalledProcessError(completed.returncode, cmd, stderr=tail)
