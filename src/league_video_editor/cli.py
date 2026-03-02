from __future__ import annotations

import argparse
from collections import deque
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
LOUDNORM_FILTER = "loudnorm=I=-14:LRA=11:TP=-1.5"
LOUDNORM_NONFINITE_PATTERN = re.compile(r"input contains.*nan.*inf", re.IGNORECASE | re.DOTALL)
LOUDNORM_ANALYSIS_FILTER = f"{LOUDNORM_FILTER}:print_format=json"
VIDEO_ENCODERS = ("libx264", "h264_videotoolbox", "h264_nvenc", "h264_qsv", "h264_amf")
DEFAULT_TARGET_DURATION_SECONDS = 600.0
DEFAULT_INTRO_SECONDS = 45.0
DEFAULT_OUTRO_SECONDS = 60.0


class EditorError(RuntimeError):
    """Raised when the editor cannot proceed."""


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


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


def _video_postprocess_filter(*, allow_upscale: bool) -> str:
    if allow_upscale:
        scale = "scale=w=1920:h=1080:force_original_aspect_ratio=decrease"
    else:
        scale = (
            "scale="
            "w='if(gt(iw,1920),1920,iw)':"
            "h='if(gt(ih,1080),1080,ih)':"
            "force_original_aspect_ratio=decrease"
        )
    return (
        f"{scale},"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,"
        "fps=60,"
        "format=yuv420p"
    )


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
    duration_text = result.stdout.strip()
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
                    print(
                        _render_progress_line("Analyzing scenes", progress_ratio),
                        end="",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_output_ratio = progress_ratio

        return_code = process.wait()
        if duration_seconds > 0:
            print(
                _render_progress_line("Analyzing scenes", 1.0),
                file=sys.stderr,
                flush=True,
            )
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
) -> list[float]:
    cmd = [
        "ffmpeg",
        "-hide_banner",
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
        )

    result = _run_command(cmd, capture_output=True)
    output = (result.stderr or "") + "\n" + (result.stdout or "")
    matches = SCENE_PTS_PATTERN.findall(output)
    return sorted({float(value) for value in matches if float(value) >= 0.0})


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
    clip_after: float,
    outro_seconds: float,
) -> Segment | None:
    if duration_seconds <= 0 or outro_seconds <= 0:
        return None

    likely_finish = duration_seconds
    if events:
        last_event = max(events)
        if duration_seconds - last_event <= 180.0:
            likely_finish = min(duration_seconds, last_event + max(clip_after, 12.0))

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
) -> tuple[list[Segment], bool]:
    if duration_seconds <= 0 or max_clips <= 0:
        return [], False

    intro_end = min(duration_seconds, max(0.0, intro_seconds))
    intro_segment = Segment(start=0.0, end=intro_end) if intro_end > 0 else None
    outro_segment = _derive_outro_segment(
        duration_seconds=duration_seconds,
        events=events,
        clip_after=clip_after,
        outro_seconds=max(0.0, outro_seconds),
    )

    anchored_segments = [segment for segment in (intro_segment, outro_segment) if segment is not None]
    if (
        intro_segment is not None
        and outro_segment is not None
        and intro_segment.end >= outro_segment.start
    ):
        return [Segment(0.0, duration_seconds)], False

    interior_start = intro_segment.end if intro_segment is not None else 0.0
    interior_end = outro_segment.start if outro_segment is not None else duration_seconds
    clip_window_seconds = clip_before + clip_after
    target_total_duration = min(duration_seconds, max(0.0, target_duration_seconds))
    anchored_duration = sum(segment.duration for segment in anchored_segments)
    target_middle_duration = max(0.0, target_total_duration - anchored_duration)

    middle_clip_target = 0
    if clip_window_seconds > 0 and interior_end > interior_start and target_middle_duration > 0:
        middle_clip_target = min(max_clips, math.ceil(target_middle_duration / clip_window_seconds))

    middle_segments: list[Segment] = []
    used_fallback = False
    if middle_clip_target > 0:
        effective_clip_before = clip_before
        effective_clip_after = clip_after
        if clip_window_seconds > 0 and target_middle_duration > 0:
            target_per_clip = target_middle_duration / middle_clip_target
            if target_per_clip > clip_window_seconds:
                expansion_factor = target_per_clip / clip_window_seconds
                effective_clip_before *= expansion_factor
                effective_clip_after *= expansion_factor

        ranked_events = _rank_event_candidates(
            events,
            min_gap_seconds=min_gap_seconds,
            duration_seconds=duration_seconds,
            clip_before=effective_clip_before,
            clip_after=effective_clip_after,
        )

        selected_centers: list[float] = []
        for event in ranked_events:
            if event <= interior_start or event >= interior_end:
                continue
            if any(abs(event - chosen) < min_gap_seconds for chosen in selected_centers):
                continue
            selected_centers.append(event)
            if len(selected_centers) >= middle_clip_target:
                break

        if len(selected_centers) < middle_clip_target:
            used_fallback = True
            selected_centers.extend(
                _generate_fallback_centers(
                    start=interior_start,
                    end=interior_end,
                    count=middle_clip_target - len(selected_centers),
                    min_gap_seconds=min_gap_seconds,
                    existing=selected_centers,
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
    return merge_segments(all_segments), used_fallback


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
    settings: dict[str, float | int],
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
            f"{_video_postprocess_filter(allow_upscale=allow_upscale)}"
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
) -> None:
    if command_without_loudnorm is None:
        _run_command(command_with_loudnorm)
        return

    try:
        _run_command_with_stderr_tail(command_with_loudnorm)
    except subprocess.CalledProcessError as error:
        if not _looks_like_loudnorm_nonfinite_error(error):
            raise
        print(
            "Warning: loudnorm failed due to non-finite audio values; retrying without loudness normalization.",
            file=sys.stderr,
        )
        _run_command(command_without_loudnorm)


def _run_command_with_stderr_tail(cmd: list[str], *, tail_bytes: int = 256 * 1024) -> None:
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
    crossfade_seconds: float,
    audio_fade_seconds: float,
    two_pass_loudnorm: bool,
) -> None:
    segments = read_plan(plan_path)
    if not segments:
        raise EditorError(
            f"Plan {plan_path} has no segments. Re-run analyze with a lower scene threshold."
        )

    include_audio = _has_audio_stream(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not include_audio:
        filter_graph, map_video, map_audio = _build_filter_complex(
            segments,
            include_audio=False,
            include_video=True,
            allow_upscale=allow_upscale,
            crossfade_seconds=crossfade_seconds,
            audio_fade_seconds=audio_fade_seconds,
        )
        if map_video is None:
            raise EditorError("Failed to build video filter graph.")
        command = [
            "ffmpeg",
            "-hide_banner",
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
        crossfade_seconds=crossfade_seconds,
        audio_fade_seconds=audio_fade_seconds,
        audio_post_filter=None,
    )
    if fallback_video_map is None or fallback_audio_map is None:
        raise EditorError("Failed to build fallback audio/video filter graph.")

    normalized_command = [
        "ffmpeg",
        "-hide_banner",
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
    _run_with_loudnorm_fallback(normalized_command, fallback_command)


def transcode_full_match(
    *,
    input_path: Path,
    output_path: Path,
    crf: int,
    preset: str,
    video_encoder: str,
    allow_upscale: bool,
    two_pass_loudnorm: bool,
) -> None:
    has_audio = _has_audio_stream(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        _video_postprocess_filter(allow_upscale=allow_upscale),
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
    _run_with_loudnorm_fallback(normalized_command, fallback_command)


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
    intro_seconds: float,
    outro_seconds: float,
) -> dict[str, int | float | bool]:
    duration_seconds = _probe_duration_seconds(input_path)
    events = detect_scene_events(
        input_path,
        scene_threshold,
        duration_seconds=duration_seconds,
        show_progress=True,
    )
    segments, used_fallback = build_segments(
        events,
        duration_seconds=duration_seconds,
        clip_before=clip_before,
        clip_after=clip_after,
        min_gap_seconds=min_gap_seconds,
        max_clips=max_clips,
        target_duration_seconds=target_duration_seconds,
        intro_seconds=intro_seconds,
        outro_seconds=outro_seconds,
    )
    settings = {
        "scene_threshold": scene_threshold,
        "clip_before": clip_before,
        "clip_after": clip_after,
        "min_gap_seconds": min_gap_seconds,
        "max_clips": max_clips,
        "target_duration_seconds": target_duration_seconds,
        "intro_seconds": intro_seconds,
        "outro_seconds": outro_seconds,
    }
    write_plan(
        input_path=input_path,
        output_path=plan_path,
        duration_seconds=duration_seconds,
        events=events,
        segments=segments,
        used_fallback=used_fallback,
        settings=settings,
    )
    return {
        "event_count": len(events),
        "segment_count": len(segments),
        "used_fallback": used_fallback,
        "duration_seconds": duration_seconds,
    }


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
        ):
            _require_finite(option_name, option_value)
            if option_value < 0:
                raise EditorError(f"--{option_name} must be >= 0.")

        if args.target_duration_seconds <= 0:
            raise EditorError("--target-duration-seconds must be > 0.")

        if args.max_clips < 1:
            raise EditorError("--max-clips must be >= 1.")

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
    analyze.add_argument("--clip-before", type=float, default=8.0)
    analyze.add_argument("--clip-after", type=float, default=12.0)
    analyze.add_argument("--min-gap-seconds", type=float, default=18.0)
    analyze.add_argument(
        "--max-clips",
        type=int,
        default=20,
        help="Maximum number of middle highlight clips between intro and outro anchors.",
    )
    analyze.add_argument(
        "--target-duration-seconds",
        type=float,
        default=DEFAULT_TARGET_DURATION_SECONDS,
        help="Approximate target duration for final highlights output.",
    )
    analyze.add_argument(
        "--intro-seconds",
        type=float,
        default=DEFAULT_INTRO_SECONDS,
        help="Seconds to keep from the beginning for game load-in.",
    )
    analyze.add_argument(
        "--outro-seconds",
        type=float,
        default=DEFAULT_OUTRO_SECONDS,
        help="Seconds to keep at the end for match finish/nexus destruction.",
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
    auto.add_argument("--clip-before", type=float, default=8.0)
    auto.add_argument("--clip-after", type=float, default=12.0)
    auto.add_argument("--min-gap-seconds", type=float, default=18.0)
    auto.add_argument(
        "--max-clips",
        type=int,
        default=20,
        help="Maximum number of middle highlight clips between intro and outro anchors.",
    )
    auto.add_argument(
        "--target-duration-seconds",
        type=float,
        default=DEFAULT_TARGET_DURATION_SECONDS,
        help="Approximate target duration for final highlights output.",
    )
    auto.add_argument(
        "--intro-seconds",
        type=float,
        default=DEFAULT_INTRO_SECONDS,
        help="Seconds to keep from the beginning for game load-in.",
    )
    auto.add_argument(
        "--outro-seconds",
        type=float,
        default=DEFAULT_OUTRO_SECONDS,
        help="Seconds to keep at the end for match finish/nexus destruction.",
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
                intro_seconds=args.intro_seconds,
                outro_seconds=args.outro_seconds,
            )
            print(
                f"Wrote {args.plan} with {stats['segment_count']} segments "
                f"from {stats['event_count']} detected events "
                f"(fallback={stats['used_fallback']})."
            )
            return 0

        if args.command == "render":
            render_highlights(
                input_path=args.input,
                plan_path=args.plan,
                output_path=args.output,
                crf=args.crf,
                preset=args.preset,
                video_encoder=args.video_encoder,
                allow_upscale=args.allow_upscale,
                crossfade_seconds=args.crossfade_seconds,
                audio_fade_seconds=args.audio_fade_seconds,
                two_pass_loudnorm=args.two_pass_loudnorm,
            )
            print(f"Rendered highlights: {args.output}")
            return 0

        if args.command == "auto":
            stats = analyze_recording(
                input_path=args.input,
                plan_path=args.plan,
                scene_threshold=args.scene_threshold,
                clip_before=args.clip_before,
                clip_after=args.clip_after,
                min_gap_seconds=args.min_gap_seconds,
                max_clips=args.max_clips,
                target_duration_seconds=args.target_duration_seconds,
                intro_seconds=args.intro_seconds,
                outro_seconds=args.outro_seconds,
            )
            render_highlights(
                input_path=args.input,
                plan_path=args.plan,
                output_path=args.output,
                crf=args.crf,
                preset=args.preset,
                video_encoder=args.video_encoder,
                allow_upscale=args.allow_upscale,
                crossfade_seconds=args.crossfade_seconds,
                audio_fade_seconds=args.audio_fade_seconds,
                two_pass_loudnorm=args.two_pass_loudnorm,
            )
            print(
                f"Rendered {args.output} using {stats['segment_count']} segments "
                f"(plan: {args.plan})."
            )
            return 0

        if args.command == "full":
            transcode_full_match(
                input_path=args.input,
                output_path=args.output,
                crf=args.crf,
                preset=args.preset,
                video_encoder=args.video_encoder,
                allow_upscale=args.allow_upscale,
                two_pass_loudnorm=args.two_pass_loudnorm,
            )
            print(f"Rendered full match: {args.output}")
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
