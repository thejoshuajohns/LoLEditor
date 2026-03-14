"""Watchability analysis: scoring and reporting for highlight reels."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from pathlib import Path

from .analyze import _compute_vision_window_scores, _percentile
from .ffmpeg_utils import (
    detect_scene_events_adaptive,
    extract_signalstats_samples,
    probe_duration_seconds,
    render_progress_line,
)
from .models import (
    DEFAULT_VISION_SAMPLE_FPS,
    DEFAULT_VISION_STEP_SECONDS,
    DEFAULT_VISION_WINDOW_SECONDS,
    DEFAULT_WATCHABILITY_SCENE_THRESHOLD,
    VisionWindow,
)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


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
# Signal detection helpers
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


# ---------------------------------------------------------------------------
# Core report builder
# ---------------------------------------------------------------------------


def _build_watchability_report(
    *,
    duration_seconds: float,
    events: list[float],
    vision_windows: list[VisionWindow],
    scene_threshold_used: float,
) -> dict[str, object]:
    duration_minutes = duration_seconds / 60 if duration_seconds > 0 else 0.0
    event_rate_per_minute = len(events) / duration_minutes if duration_minutes > 0 else 0.0

    scores = [w.score for w in vision_windows]
    saturations = [w.saturation for w in vision_windows]
    avg_score = _average(scores)
    score_std = _stddev(scores)
    low_activity_threshold = _percentile(scores, 0.30)
    high_activity_threshold = _percentile(scores, 0.75)
    low_activity_ratio = (
        sum(1 for s in scores if s <= low_activity_threshold) / len(scores)
        if scores
        else 1.0
    )
    high_activity_ratio = (
        sum(1 for s in scores if s >= high_activity_threshold) / len(scores)
        if scores
        else 0.0
    )

    ordered = sorted(vision_windows, key=lambda w: w.start)
    death_cues = _detect_death_cues(ordered, min_spacing_seconds=20.0)
    motions = [w.motion for w in ordered]
    low_motion_threshold = _percentile(motions, 0.45)
    low_score_threshold = _percentile(scores, 0.50)
    low_sat_threshold = _percentile(saturations, 0.30)
    avg_window_duration = _average([w.end - w.start for w in ordered])
    death_context_seconds = max(12.0, avg_window_duration * 1.5)
    death_like_windows = 0
    for w in ordered:
        if not death_cues:
            break
        center = (w.start + w.end) / 2
        near_death = any(abs(center - t) <= death_context_seconds for t in death_cues)
        if not near_death:
            continue
        if (
            w.motion <= low_motion_threshold
            and w.score <= low_score_threshold
            and w.saturation <= low_sat_threshold
        ):
            death_like_windows += 1
    gray_ratio = death_like_windows / len(ordered) if ordered else 0.0

    expected_event_rate = min(
        14.0,
        max(0.8, 0.9 + 7.5 * avg_score + 3.0 * high_activity_ratio),
    )
    if avg_score < 0.34 and low_activity_ratio > 0.50:
        activity_profile = "low-action"
        expected_event_rate = min(expected_event_rate, 3.0)
    elif avg_score > 0.62 or high_activity_ratio > 0.30:
        activity_profile = "high-action"
    else:
        activity_profile = "standard"

    pacing_tolerance = max(2.0, expected_event_rate * 0.9)
    raw_pacing = _closeness_score(
        event_rate_per_minute,
        target=expected_event_rate,
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

    quality_target = {
        "low-action": 1.6,
        "standard": 2.5,
        "high-action": 3.6,
    }[activity_profile]
    quality_tolerance = max(0.9, quality_target * 0.55)
    quality_pacing_raw = _closeness_score(
        event_rate_per_minute,
        target=quality_target,
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
        watchability_weight, quality_weight = 0.30, 0.70
    elif activity_profile == "high-action":
        watchability_weight, quality_weight = 0.55, 0.45
    else:
        watchability_weight, quality_weight = 0.45, 0.55
    youtube_score = min(
        100.0,
        max(0.0, overall_score * watchability_weight + highlight_quality_score * quality_weight),
    )

    recommendations: list[str] = []
    if low_activity_ratio > 0.45:
        recommendations.append("Trim or speed up extended low-activity windows.")
    if event_rate_per_minute > 18:
        recommendations.append("Reduce cut frequency to improve flow continuity.")
    low_pace_threshold = max(1.2, expected_event_rate * 0.45)
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
        if event_rate_per_minute < quality_target:
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
        r for r in quality_recommendations if r not in recommendations
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
        "expected_event_rate_per_minute": round(expected_event_rate, 3),
        "avg_activity_score": round(avg_score, 4),
        "activity_score_stddev": round(score_std, 4),
        "low_activity_ratio": round(low_activity_ratio, 4),
        "high_activity_ratio": round(high_activity_ratio, 4),
        "gray_screen_ratio": round(gray_ratio, 4),
        "quality_recommendations": quality_recommendations,
        "recommendations": merged_recommendations,
    }


# ---------------------------------------------------------------------------
# Public entry point
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

    duration_seconds = probe_duration_seconds(input_path)
    if show_progress:
        emit_progress(0.05)

    events, scene_threshold_used = detect_scene_events_adaptive(
        input_path,
        scene_threshold,
        duration_seconds=duration_seconds,
        show_progress=show_progress,
        progress_label="Analyzing scenes",
        progress_callback=stage_callback(0.05, 0.55) if show_progress else None,
    )
    if show_progress:
        emit_progress(0.55)

    samples = extract_signalstats_samples(
        input_path,
        duration_seconds=duration_seconds,
        sample_fps=vision_sample_fps,
        show_progress=show_progress,
        progress_label="Scoring gameplay",
        progress_callback=stage_callback(0.55, 0.95) if show_progress else None,
    )
    vision_windows = _compute_vision_window_scores(
        frame_samples=samples,
        events=events,
        duration_seconds=duration_seconds,
        window_seconds=vision_window_seconds,
        step_seconds=vision_step_seconds,
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
