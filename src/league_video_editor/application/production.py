"""High-level production pipeline for branded highlight packages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config.settings import ProjectConfig
from ..ffmpeg_utils import resolve_encoder, run_command, video_codec_args
from ..models import ENCODER_PROFILES
from ..render import render_highlights
from ..thumbnail import generate_thumbnail
from .overlays import build_overlay_bundle


def produce_highlight_package(
    *,
    input_path: Path,
    plan_path: Path,
    project_config: ProjectConfig,
    output_dir: Path,
    render_profile: str = "balanced",
    crf: int | None = None,
    preset: str | None = None,
    auto_crop: bool = True,
    crossfade_seconds: float = 0.6,
) -> dict[str, Any]:
    """Render highlights, apply branding, and generate final assets."""

    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    base_render_path = artifacts_dir / "base-highlights.mp4"
    branded_path = artifacts_dir / "branded-highlights.mp4"
    final_output_path = output_dir / "highlights.mp4"
    thumbnail_path = output_dir / "thumbnail.jpg"

    profile_settings = ENCODER_PROFILES.get(render_profile, ENCODER_PROFILES["balanced"])
    video_encoder = resolve_encoder(str(project_config.output.codec or profile_settings.get("video_encoder", "auto")))
    resolved_crf = int(crf if crf is not None else profile_settings.get("crf", project_config.output.crf))
    resolved_preset = str(preset or profile_settings.get("preset", project_config.output.preset))

    render_highlights(
        input_path=input_path,
        plan_path=plan_path,
        output_path=base_render_path,
        crf=resolved_crf,
        preset=resolved_preset,
        video_encoder=video_encoder,
        allow_upscale=False,
        auto_crop=auto_crop,
        crossfade_seconds=crossfade_seconds,
        audio_fade_seconds=0.3,
        two_pass_loudnorm=render_profile == "quality",
    )

    overlay_manifest = build_overlay_bundle(
        champion=project_config.champion,
        content=project_config.content,
        overlay_config=project_config.overlays,
        upload=project_config.upload,
        output_dir=output_dir,
    )

    branded_source = _apply_persistent_overlays(
        input_path=base_render_path,
        output_path=branded_path,
        overlay_manifest=overlay_manifest,
        video_encoder=video_encoder,
        crf=resolved_crf,
        preset=resolved_preset,
    )
    _compose_final_video(
        branded_video_path=branded_source,
        final_output_path=final_output_path,
        overlay_manifest=overlay_manifest,
    )

    generate_thumbnail(
        input_path=final_output_path,
        output_path=thumbnail_path,
        timestamp_seconds=None,
        auto_crop=auto_crop,
        enhance=True,
        champion_overlay_path=(
            Path(project_config.champion.champion_png)
            if project_config.champion.champion_png.strip()
            else None
        ),
        champion_scale=0.58,
        champion_anchor="right",
        headline_text=project_config.content.thumbnail_headline.strip() or None,
        headline_size=118,
        headline_color="#F4D35E",
        headline_font=None,
        headline_y_ratio=0.07,
        scene_threshold=0.2,
        vision_sample_fps=0.75,
        vision_window_seconds=8.0,
        vision_step_seconds=4.0,
        width=1280,
        height=720,
        quality=2,
    )

    artifacts = {
        "base_render": str(base_render_path),
        "branded_render": str(branded_source),
        "final_video": str(final_output_path),
        "thumbnail": str(thumbnail_path),
        "overlay_manifest": overlay_manifest,
    }
    (artifacts_dir / "artifacts.json").write_text(
        json.dumps(artifacts, indent=2) + "\n",
        encoding="utf-8",
    )
    return artifacts


def _apply_persistent_overlays(
    *,
    input_path: Path,
    output_path: Path,
    overlay_manifest: dict[str, Any],
    video_encoder: str,
    crf: int,
    preset: str,
) -> Path:
    champion_path = overlay_manifest.get("champion_portrait")
    kda_path = overlay_manifest.get("kda_overlay")
    if not champion_path and not kda_path:
        return input_path

    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", str(input_path)]
    filter_parts: list[str] = []
    current = "0:v"
    input_index = 1

    if champion_path:
        cmd.extend(["-i", str(champion_path)])
        filter_parts.append(
            f"[{input_index}:v]scale=220:-1:force_original_aspect_ratio=decrease[champ];"
            f"[{current}][champ]overlay=20:main_h-overlay_h-20:format=auto:enable='gte(t,0.5)'[v{input_index}]"
        )
        current = f"v{input_index}"
        input_index += 1

    if kda_path:
        cmd.extend(["-i", str(kda_path)])
        filter_parts.append(
            f"[{input_index}:v]format=rgba[kda];"
            f"[{current}][kda]overlay=main_w-overlay_w-20:20:format=auto:enable='gte(t,0.8)'[v{input_index}]"
        )
        current = f"v{input_index}"

    cmd.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            f"[{current}]",
            "-map",
            "0:a?",
            *video_codec_args(video_encoder=video_encoder, crf=crf, preset=preset),
            "-c:a",
            "copy",
            str(output_path),
        ]
    )
    run_command(cmd)
    return output_path


def _compose_final_video(
    *,
    branded_video_path: Path,
    final_output_path: Path,
    overlay_manifest: dict[str, Any],
) -> None:
    intro_card = overlay_manifest.get("intro_card")
    end_card = overlay_manifest.get("end_card")
    if not intro_card and not end_card:
        if branded_video_path != final_output_path:
            run_command(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-y",
                    "-i",
                    str(branded_video_path),
                    "-c",
                    "copy",
                    str(final_output_path),
                ]
            )
        return

    temp_inputs: list[Path] = []
    if intro_card:
        temp_inputs.append(_render_card_video(Path(intro_card), final_output_path.parent / "artifacts" / "intro.mp4", duration_seconds=3.8))
    temp_inputs.append(branded_video_path)
    if end_card:
        temp_inputs.append(_render_card_video(Path(end_card), final_output_path.parent / "artifacts" / "outro.mp4", duration_seconds=5.0))

    cmd = ["ffmpeg", "-hide_banner", "-y"]
    for path in temp_inputs:
        cmd.extend(["-i", str(path)])

    concat_inputs = "".join(f"[{index}:v][{index}:a]" for index in range(len(temp_inputs)))
    cmd.extend(
        [
            "-filter_complex",
            f"{concat_inputs}concat=n={len(temp_inputs)}:v=1:a=1[outv][outa]",
            "-map",
            "[outv]",
            "-map",
            "[outa]",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(final_output_path),
        ]
    )
    run_command(cmd)


def _render_card_video(
    image_path: Path,
    output_path: Path,
    *,
    duration_seconds: float,
) -> Path:
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-loop",
            "1",
            "-i",
            str(image_path),
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo",
            "-t",
            f"{duration_seconds:.2f}",
            "-vf",
            "fps=60,scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,format=yuv420p",
            "-shortest",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(output_path),
        ]
    )
    return output_path
