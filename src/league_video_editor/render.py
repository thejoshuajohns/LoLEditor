"""Video rendering: highlights montage and full-match transcode."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .analyze import read_plan
from .ffmpeg_utils import (
    build_filter_complex,
    detect_crop_filter,
    estimate_render_duration,
    has_audio_stream,
    probe_duration_seconds,
    run_command,
    run_with_loudnorm_fallback,
    select_loudnorm_filter,
    video_codec_args,
    video_postprocess_filter,
)
from .models import LOUDNORM_ANALYSIS_FILTER, EditorError


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
    _read_plan_fn=None,
    _has_audio_fn=None,
    _run_cmd_fn=None,
    _run_with_loudnorm_fn=None,
) -> None:
    _do_read_plan = _read_plan_fn if _read_plan_fn is not None else read_plan
    _do_has_audio = _has_audio_fn if _has_audio_fn is not None else has_audio_stream
    _do_run = _run_cmd_fn if _run_cmd_fn is not None else run_command
    _do_loudnorm = _run_with_loudnorm_fn if _run_with_loudnorm_fn is not None else run_with_loudnorm_fallback

    segments = _do_read_plan(plan_path)
    if not segments:
        raise EditorError(
            f"Plan {plan_path} has no segments. Re-run analyze with a lower scene threshold."
        )
    estimated_duration_seconds = estimate_render_duration(
        segments,
        crossfade_seconds=crossfade_seconds,
    )
    include_audio = _do_has_audio(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    crop_filter: str | None = None
    if auto_crop:
        try:
            duration_seconds = probe_duration_seconds(input_path)
            crop_filter = detect_crop_filter(input_path, duration_seconds=duration_seconds)
        except (EditorError, subprocess.CalledProcessError):
            crop_filter = None

    if not include_audio:
        filter_graph, map_video, _ = build_filter_complex(
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
        _do_run(
            [
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
                *video_codec_args(video_encoder=video_encoder, crf=crf, preset=preset),
                "-movflags",
                "+faststart",
                "-an",
                str(output_path),
            ]
        )
        return

    audio_analysis_command: list[str] | None = None
    if two_pass_loudnorm:
        analysis_fg, _, analysis_audio_map = build_filter_complex(
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
                analysis_fg,
                "-map",
                analysis_audio_map,
                "-f",
                "null",
                "-",
            ]

    loudnorm_filter = select_loudnorm_filter(
        two_pass_loudnorm=two_pass_loudnorm,
        analysis_command=audio_analysis_command,
        _run_cmd_fn=_do_run,
    )

    norm_fg, map_video, map_audio = build_filter_complex(
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

    fallback_fg, fb_video, fb_audio = build_filter_complex(
        segments,
        include_audio=True,
        include_video=True,
        allow_upscale=allow_upscale,
        crop_filter=crop_filter,
        crossfade_seconds=crossfade_seconds,
        audio_fade_seconds=audio_fade_seconds,
        audio_post_filter=None,
    )
    if fb_video is None or fb_audio is None:
        raise EditorError("Failed to build fallback filter graph.")

    audio_args = ["-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "48000"]
    codec_args = video_codec_args(video_encoder=video_encoder, crf=crf, preset=preset)

    def _render_cmd(fg: str, vm: str, am: str) -> list[str]:
        return [
            "ffmpeg",
            "-hide_banner",
            "-stats_period",
            "0.25",
            "-y",
            "-i",
            str(input_path),
            "-filter_complex",
            fg,
            "-map",
            vm,
            *codec_args,
            "-movflags",
            "+faststart",
            "-map",
            am,
            *audio_args,
            str(output_path),
        ]

    _do_loudnorm(
        _render_cmd(norm_fg, map_video, map_audio),
        _render_cmd(fallback_fg, fb_video, fb_audio),
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
    _has_audio_fn=None,
    _run_cmd_fn=None,
    _run_with_loudnorm_fn=None,
) -> None:
    _do_has_audio = _has_audio_fn if _has_audio_fn is not None else has_audio_stream
    _do_run = _run_cmd_fn if _run_cmd_fn is not None else run_command
    _do_loudnorm = _run_with_loudnorm_fn if _run_with_loudnorm_fn is not None else run_with_loudnorm_fallback

    include_audio = _do_has_audio(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    crop_filter: str | None = None
    duration_seconds: float | None = None
    try:
        duration_seconds = probe_duration_seconds(input_path, _run_cmd_fn=_do_run)
    except (EditorError, subprocess.CalledProcessError):
        duration_seconds = None
    if auto_crop and duration_seconds:
        try:
            crop_filter = detect_crop_filter(input_path, duration_seconds=duration_seconds)
        except (EditorError, subprocess.CalledProcessError):
            crop_filter = None

    base_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-stats_period",
        "0.25",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        video_postprocess_filter(allow_upscale=allow_upscale, crop_filter=crop_filter),
        *video_codec_args(video_encoder=video_encoder, crf=crf, preset=preset),
        "-movflags",
        "+faststart",
    ]

    if not include_audio:
        _do_run(base_cmd + ["-an", str(output_path)])
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

    loudnorm_filter = select_loudnorm_filter(
        two_pass_loudnorm=two_pass_loudnorm,
        analysis_command=analysis_command,
        _run_cmd_fn=_do_run,
    )
    audio_args = ["-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "48000"]
    normalized_cmd = base_cmd + ["-af", loudnorm_filter] + audio_args + [str(output_path)]
    fallback_cmd = base_cmd + audio_args + [str(output_path)]
    _do_loudnorm(
        normalized_cmd,
        fallback_cmd,
        progress_label="Rendering full match",
        progress_duration_seconds=duration_seconds,
    )
