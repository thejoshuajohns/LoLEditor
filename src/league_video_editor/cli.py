from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


SCENE_PTS_PATTERN = re.compile(r"pts_time:(-?\d+(?:\.\d+)?)")


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
    return float(duration_text)


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


def detect_scene_events(input_path: Path, scene_threshold: float) -> list[float]:
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
    result = subprocess.run(cmd, text=True, capture_output=True)
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


def build_segments(
    events: list[float],
    *,
    duration_seconds: float,
    clip_before: float,
    clip_after: float,
    min_gap_seconds: float,
    max_clips: int,
) -> tuple[list[Segment], bool]:
    if duration_seconds <= 0:
        return [], False

    filtered: list[float] = []
    for event in sorted(events):
        if not filtered or event - filtered[-1] >= min_gap_seconds:
            filtered.append(event)
    filtered = sample_evenly(filtered, max_clips)

    used_fallback = False
    if not filtered:
        fallback_points = [0.25, 0.5, 0.75]
        filtered = [duration_seconds * point for point in fallback_points]
        used_fallback = True

    segments: list[Segment] = []
    for event in filtered:
        start = max(0.0, event - clip_before)
        end = min(duration_seconds, event + clip_after)
        if end > start:
            segments.append(Segment(start=start, end=end))

    return merge_segments(segments), used_fallback


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
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_plan(plan_path: Path) -> list[Segment]:
    data = json.loads(plan_path.read_text(encoding="utf-8"))
    segments: list[Segment] = []
    for raw in data.get("segments", []):
        start = float(raw["start"])
        end = float(raw["end"])
        if end > start:
            segments.append(Segment(start=start, end=end))
    return segments


def _build_filter_complex(segments: list[Segment], include_audio: bool) -> tuple[str, str, str | None]:
    pieces: list[str] = []
    for index, segment in enumerate(segments):
        pieces.append(
            f"[0:v]trim=start={segment.start:.3f}:end={segment.end:.3f},setpts=PTS-STARTPTS[v{index}]"
        )
        if include_audio:
            pieces.append(
                f"[0:a]atrim=start={segment.start:.3f}:end={segment.end:.3f},asetpts=PTS-STARTPTS[a{index}]"
            )

    if include_audio:
        concat_inputs = "".join(f"[v{index}][a{index}]" for index in range(len(segments)))
        pieces.append(
            f"{concat_inputs}concat=n={len(segments)}:v=1:a=1[vcat][acat]"
        )
        pieces.append(
            "[vcat]scale=w=1920:h=1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,fps=60,format=yuv420p[vout]"
        )
        pieces.append("[acat]loudnorm=I=-14:LRA=11:TP=-1.5[aout]")
        return ";".join(pieces), "[vout]", "[aout]"

    concat_inputs = "".join(f"[v{index}]" for index in range(len(segments)))
    pieces.append(f"{concat_inputs}concat=n={len(segments)}:v=1:a=0[vcat]")
    pieces.append(
        "[vcat]scale=w=1920:h=1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,fps=60,format=yuv420p[vout]"
    )
    return ";".join(pieces), "[vout]", None


def render_highlights(
    *,
    input_path: Path,
    plan_path: Path,
    output_path: Path,
    crf: int,
    preset: str,
) -> None:
    segments = read_plan(plan_path)
    if not segments:
        raise EditorError(
            f"Plan {plan_path} has no segments. Re-run analyze with a lower scene threshold."
        )

    include_audio = _has_audio_stream(input_path)
    filter_graph, map_video, map_audio = _build_filter_complex(segments, include_audio)

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
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if map_audio is not None:
        command.extend(
            [
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
            ]
        )
    else:
        command.append("-an")

    command.append(str(output_path))
    _run_command(command)


def transcode_full_match(
    *,
    input_path: Path,
    output_path: Path,
    crf: int,
    preset: str,
) -> None:
    has_audio = _has_audio_stream(input_path)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        "scale=w=1920:h=1080:force_original_aspect_ratio=decrease,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,fps=60,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
    ]
    if has_audio:
        command.extend(
            [
                "-af",
                "loudnorm=I=-14:LRA=11:TP=-1.5",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ac",
                "2",
                "-ar",
                "48000",
            ]
        )
    else:
        command.append("-an")
    command.append(str(output_path))
    _run_command(command)


def analyze_recording(
    *,
    input_path: Path,
    plan_path: Path,
    scene_threshold: float,
    clip_before: float,
    clip_after: float,
    min_gap_seconds: float,
    max_clips: int,
) -> dict[str, int | float | bool]:
    duration_seconds = _probe_duration_seconds(input_path)
    events = detect_scene_events(input_path, scene_threshold)
    segments, used_fallback = build_segments(
        events,
        duration_seconds=duration_seconds,
        clip_before=clip_before,
        clip_after=clip_after,
        min_gap_seconds=min_gap_seconds,
        max_clips=max_clips,
    )
    settings = {
        "scene_threshold": scene_threshold,
        "clip_before": clip_before,
        "clip_after": clip_after,
        "min_gap_seconds": min_gap_seconds,
        "max_clips": max_clips,
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
    analyze.add_argument("--max-clips", type=int, default=20)

    render = subparsers.add_parser(
        "render", help="Render highlights MP4 from an existing edit plan JSON."
    )
    render.add_argument("input", type=Path, help="Path to OBS recording")
    render.add_argument("--plan", type=Path, required=True, help="Path to edit plan JSON")
    render.add_argument("--output", type=Path, default=Path("highlights.mp4"))
    render.add_argument("--crf", type=int, default=20)
    render.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
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
    auto.add_argument("--max-clips", type=int, default=20)
    auto.add_argument("--crf", type=int, default=20)
    auto.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
    )

    full = subparsers.add_parser(
        "full",
        help="Transcode full match to YouTube-ready format without clipping highlights.",
    )
    full.add_argument("input", type=Path, help="Path to OBS recording")
    full.add_argument("--output", type=Path, default=Path("full-match-youtube.mp4"))
    full.add_argument("--crf", type=int, default=20)
    full.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
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
            )
            render_highlights(
                input_path=args.input,
                plan_path=args.plan,
                output_path=args.output,
                crf=args.crf,
                preset=args.preset,
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
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

