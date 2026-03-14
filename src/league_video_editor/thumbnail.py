"""YouTube thumbnail generator."""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

from .analyze import (
    _compute_vision_window_scores,
    _normalize,
    _percentile,
    score_transcript_text,
)
from .ffmpeg_utils import (
    detect_crop_filter,
    detect_scene_events_adaptive,
    extract_signalstats_samples,
    probe_duration_seconds,
    run_command,
)
from .models import (
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
    EditorError,
    VisionWindow,
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
        max_ow = max(96, int(round(width * champion_scale)))
        max_oh = max(96, int(round(height * 0.96)))
        lines.append(
            f"[{champion_input_index}:v]"
            f"scale=w={max_ow}:h={max_oh}:force_original_aspect_ratio=decrease"
            "[champ]"
        )
        if champion_anchor == "left":
            x_expr = "24"
        elif champion_anchor == "center":
            x_expr = "(main_w-overlay_w)/2"
        else:
            x_expr = "main_w-overlay_w-24"
        lines.append(f"[{current}][champ]overlay=x={x_expr}:y=main_h-overlay_h-14:format=auto[thumb_overlay]")
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
    except ImportError as err:
        raise EditorError(
            "Headline text overlay requires Pillow. Run: python3 -m pip install pillow"
        ) from err

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
    except ValueError as err:
        raise EditorError(f"Invalid --headline-color value: {headline_color!r}") from err

    stroke_width = max(2, int(round(headline_size * 0.065)))
    line_spacing = max(4, int(round(headline_size * 0.16)))
    lines = [line for line in headline_text.splitlines() if line.strip()] or [headline_text.strip()]
    measurements: list[tuple[str, int, int]] = []
    max_width = total_height = 0
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_width)
        lw = max(0, bbox[2] - bbox[0])
        lh = max(0, bbox[3] - bbox[1])
        measurements.append((line, lw, lh))
        max_width = max(max_width, lw)
        total_height += lh
        if i + 1 < len(lines):
            total_height += line_spacing

    top = int(height * max(0.0, min(1.0, headline_y_ratio)))
    top = min(max(8, top), max(8, height - total_height - 16))
    y = top
    fill = (rgb[0], rgb[1], rgb[2], 255)
    for line, lw, lh in measurements:
        x = max(0, (width - lw) // 2)
        draw.text((x, y), line, font=font, fill=fill, stroke_width=stroke_width, stroke_fill=(0, 0, 0, 230))
        y += lh + line_spacing
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
        w for w in vision_windows
        if w.end > w.start and w.end >= safe_start and w.start <= safe_end
    ] or [w for w in vision_windows if w.end > w.start]
    if not candidates:
        return fallback

    motions = [w.motion for w in candidates if math.isfinite(w.motion)]
    saturations = [w.saturation for w in candidates if math.isfinite(w.saturation)]
    densities = [w.scene_density for w in candidates if math.isfinite(w.scene_density)]
    motion_low = _percentile(motions, 0.15)
    motion_high = _percentile(motions, 0.90)
    sat_low = _percentile(saturations, 0.15)
    sat_high = _percentile(saturations, 0.90)
    density_low = _percentile(densities, 0.10)
    density_high = _percentile(densities, 0.90)

    best_score = -1.0
    best_ts = fallback
    for w in candidates:
        center = (w.start + w.end) / 2.0
        clamped = min(safe_end, max(safe_start, center))
        ratio = clamped / duration_seconds if duration_seconds > 0 else 0.5
        midpoint_bias = 1.0 - min(1.0, abs(ratio - 0.55) / 0.55)
        score = (
            0.62 * max(0.0, min(1.0, w.score))
            + 0.14 * _normalize(w.motion, low=motion_low, high=motion_high)
            + 0.10 * _normalize(w.scene_density, low=density_low, high=density_high)
            + 0.10 * _normalize(w.saturation, low=sat_low, high=sat_high)
            + 0.04 * midpoint_bias
        )
        if score > best_score:
            best_score = score
            best_ts = clamped
    return best_ts


def generate_thumbnail(
    *,
    input_path: Path,
    output_path: Path,
    timestamp_seconds: float | None,
    scene_threshold: float = DEFAULT_THUMBNAIL_SCENE_THRESHOLD,
    vision_sample_fps: float = DEFAULT_THUMBNAIL_VISION_SAMPLE_FPS,
    vision_window_seconds: float = DEFAULT_THUMBNAIL_VISION_WINDOW_SECONDS,
    vision_step_seconds: float = DEFAULT_THUMBNAIL_VISION_STEP_SECONDS,
    width: int = DEFAULT_THUMBNAIL_WIDTH,
    height: int = DEFAULT_THUMBNAIL_HEIGHT,
    quality: int = DEFAULT_THUMBNAIL_QUALITY,
    auto_crop: bool = True,
    enhance: bool = True,
    champion_overlay_path: Path | None = None,
    champion_scale: float = DEFAULT_THUMBNAIL_CHAMPION_SCALE,
    champion_anchor: str = DEFAULT_THUMBNAIL_CHAMPION_ANCHOR,
    headline_text: str | None = None,
    headline_size: int = DEFAULT_THUMBNAIL_HEADLINE_SIZE,
    headline_color: str = DEFAULT_THUMBNAIL_HEADLINE_COLOR,
    headline_font: Path | None = None,
    headline_y_ratio: float = DEFAULT_THUMBNAIL_HEADLINE_Y_RATIO,
    _probe_fn=None,
    _detect_crop_fn=None,
    _create_headline_fn=None,
    _run_cmd_fn=None,
    _detect_events_fn=None,
    _score_vision_fn=None,
) -> dict[str, float | int | bool]:
    _do_probe = _probe_fn if _probe_fn is not None else probe_duration_seconds
    _do_crop = _detect_crop_fn if _detect_crop_fn is not None else detect_crop_filter
    _do_headline = _create_headline_fn if _create_headline_fn is not None else _create_headline_overlay_image
    _do_run = _run_cmd_fn if _run_cmd_fn is not None else run_command
    _do_detect_events = _detect_events_fn if _detect_events_fn is not None else detect_scene_events_adaptive
    duration_seconds = _do_probe(input_path)
    auto_selected = timestamp_seconds is None
    scene_threshold_used = scene_threshold
    events: list[float] = []
    vision_windows: list[VisionWindow] = []

    if auto_selected:
        events, scene_threshold_used = _do_detect_events(
            input_path,
            scene_threshold,
            duration_seconds=duration_seconds,
            show_progress=True,
            progress_label="Analyzing scenes",
        )
        if _score_vision_fn is not None:
            vision_windows = _score_vision_fn(
                input_path,
                duration_seconds=duration_seconds,
                sample_fps=vision_sample_fps,
                window_seconds=vision_window_seconds,
                step_seconds=vision_step_seconds,
                events=events,
                show_progress=True,
                progress_label="Scoring gameplay",
            )
        else:
            samples = extract_signalstats_samples(
                input_path,
                duration_seconds=duration_seconds,
                sample_fps=vision_sample_fps,
                show_progress=True,
                progress_label="Scoring gameplay",
            )
            vision_windows = _compute_vision_window_scores(
                frame_samples=samples,
                events=events,
                duration_seconds=duration_seconds,
                window_seconds=vision_window_seconds,
                step_seconds=vision_step_seconds,
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
            crop_filter = _do_crop(input_path, duration_seconds=duration_seconds)
        except (EditorError, subprocess.CalledProcessError):
            crop_filter = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_headline = None
    if headline_text is not None and headline_text.strip():
        normalized_headline = headline_text.replace("\\n", "\n").strip()

    with tempfile.TemporaryDirectory(prefix="lol-thumb-") as tmp:
        headline_path: Path | None = None
        if normalized_headline:
            headline_path = Path(tmp) / "headline.png"
            _do_headline(
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
        next_idx = 1
        if champion_overlay_path is not None:
            champion_input_index = next_idx
            next_idx += 1
        if headline_path is not None:
            headline_input_index = next_idx

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
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-ss",
            f"{selected_timestamp:.3f}",
            "-i",
            str(input_path),
        ]
        if champion_overlay_path is not None:
            cmd.extend(["-i", str(champion_overlay_path)])
        if headline_path is not None:
            cmd.extend(["-i", str(headline_path)])
        cmd.extend(
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
            cmd.extend(["-q:v", str(quality)])
        cmd.append(str(output_path))
        _do_run(cmd, capture_output=True)

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
