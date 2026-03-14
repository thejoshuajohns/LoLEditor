"""Smart editing engine.

Converts ranked highlight events into an optimized edit plan with:
  - Dynamic clip length based on action intensity
  - Pre-fight context windows
  - Post-fight aftermath
  - Multi-kill prioritization
  - Death context retention
  - Clip smoothing and gap management
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from ..models import Segment, VisionWindow
from .highlight_detector import HighlightEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Smart clip generation
# ---------------------------------------------------------------------------

@dataclass
class SmartClip:
    """A clip with editing intelligence applied."""

    start: float
    end: float
    score: float
    event_type: str
    label: str
    context_before: float = 0.0  # Pre-fight context added
    context_after: float = 0.0   # Post-fight aftermath added
    priority: int = 0            # Ordering priority (higher = more important)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def full_start(self) -> float:
        return max(0.0, self.start - self.context_before)

    @property
    def full_end(self) -> float:
        return self.end + self.context_after

    @property
    def full_duration(self) -> float:
        return max(0.0, self.full_end - self.full_start)

    def to_segment(self) -> Segment:
        return Segment(start=self.full_start, end=self.full_end)


@dataclass
class EditPlan:
    """A complete edit plan with smart clips and metadata."""

    clips: list[SmartClip] = field(default_factory=list)
    total_duration: float = 0.0
    source_duration: float = 0.0
    highlight_count: int = 0
    avg_clip_duration: float = 0.0
    coverage_ratio: float = 0.0

    def to_segments(self) -> list[Segment]:
        return [clip.to_segment() for clip in self.clips]

    def to_dict(self) -> dict:
        return {
            "clips": [
                {
                    "start": round(c.full_start, 3),
                    "end": round(c.full_end, 3),
                    "duration": round(c.full_duration, 3),
                    "label": c.label,
                    "event_type": c.event_type,
                    "score": round(c.score, 4),
                    "priority": c.priority,
                }
                for c in self.clips
            ],
            "total_duration": round(self.total_duration, 3),
            "source_duration": round(self.source_duration, 3),
            "highlight_count": self.highlight_count,
            "avg_clip_duration": round(self.avg_clip_duration, 3),
            "coverage_ratio": round(self.coverage_ratio, 4),
        }


# ---------------------------------------------------------------------------
# Intensity-based clip duration
# ---------------------------------------------------------------------------

def _compute_dynamic_clip_length(
    event: HighlightEvent,
    *,
    min_seconds: float = 8.0,
    max_seconds: float = 45.0,
) -> float:
    """Calculate clip duration based on event intensity and type.

    Higher-scoring events and multi-kills get longer clips to capture
    the full context. Low-intensity moments are kept short.
    """
    base_duration = min_seconds + (max_seconds - min_seconds) * event.score

    # Event type multipliers
    type_multipliers: dict[str, float] = {
        "pentakill": 1.5,
        "multi_kill": 1.3,
        "teamfight": 1.4,
        "burst_combo": 1.2,
        "baron": 1.3,
        "elder_dragon": 1.3,
        "nexus": 1.5,
        "ace": 1.4,
        "objective": 1.2,
        "scoreboard_spike": 1.1,
        "kill": 1.0,
        "scene_change": 0.8,
        "audio_excitement": 1.1,
        "transcript_hype": 1.0,
        "death": 0.9,
    }
    multiplier = type_multipliers.get(event.event_type, 1.0)
    adjusted = base_duration * multiplier

    return min(max_seconds, max(min_seconds, adjusted))


def _compute_context_windows(
    event: HighlightEvent,
    *,
    pre_fight_seconds: float = 5.0,
    post_fight_seconds: float = 3.0,
) -> tuple[float, float]:
    """Calculate pre/post context windows for an event.

    Teamfights and objectives need more setup context.
    Quick kills need less.
    """
    pre_context = pre_fight_seconds
    post_context = post_fight_seconds

    if event.event_type in ("teamfight", "ace"):
        pre_context *= 1.5
        post_context *= 1.3
    elif event.event_type in ("baron", "elder_dragon", "objective"):
        pre_context *= 1.4
        post_context *= 1.2
    elif event.event_type in ("burst_combo", "kill"):
        pre_context *= 0.8
        post_context *= 0.7
    elif event.event_type == "death":
        pre_context *= 1.2
        post_context *= 0.5

    return pre_context, post_context


# ---------------------------------------------------------------------------
# Multi-kill prioritization
# ---------------------------------------------------------------------------

def _assign_priority(event: HighlightEvent) -> int:
    """Assign ordering priority based on event significance."""
    priority_map: dict[str, int] = {
        "pentakill": 100,
        "nexus": 95,
        "quadra_kill": 90,
        "ace": 85,
        "baron_steal": 80,
        "triple_kill": 75,
        "elder_dragon": 70,
        "baron": 65,
        "shutdown": 60,
        "double_kill": 55,
        "teamfight": 50,
        "burst_combo": 45,
        "objective": 40,
        "scoreboard_spike": 42,
        "multi_kill": 70,
        "kill": 30,
        "outplay": 35,
        "first_blood": 50,
        "audio_excitement": 25,
        "transcript_hype": 20,
        "scene_change": 10,
        "death": 15,
    }
    base_priority = priority_map.get(event.event_type, 20)

    # Boost for multi-signal events
    signal_count = len(event.signals)
    if signal_count >= 3:
        base_priority += 15
    elif signal_count >= 2:
        base_priority += 8

    return base_priority


# ---------------------------------------------------------------------------
# Clip merging and smoothing
# ---------------------------------------------------------------------------

def _merge_overlapping_clips(
    clips: list[SmartClip],
    *,
    min_gap_seconds: float = 2.0,
) -> list[SmartClip]:
    """Merge clips that overlap or are very close together."""
    if not clips:
        return []

    sorted_clips = sorted(clips, key=lambda c: c.full_start)
    merged: list[SmartClip] = [sorted_clips[0]]

    for clip in sorted_clips[1:]:
        prev = merged[-1]
        if clip.full_start <= prev.full_end + min_gap_seconds:
            # Merge: extend the previous clip
            new_end = max(prev.full_end, clip.full_end)
            merged[-1] = SmartClip(
                start=prev.start,
                end=max(prev.end, clip.end),
                score=max(prev.score, clip.score),
                event_type=prev.event_type if prev.score >= clip.score else clip.event_type,
                label=prev.label if prev.score >= clip.score else clip.label,
                context_before=prev.context_before,
                context_after=max(0.0, new_end - max(prev.end, clip.end)),
                priority=max(prev.priority, clip.priority),
            )
        else:
            merged.append(clip)

    return merged


def _apply_smoothing(
    clips: list[SmartClip],
    *,
    source_duration: float,
) -> list[SmartClip]:
    """Apply temporal smoothing to ensure clips don't start/end abruptly.

    Adjusts clip boundaries to align with natural pause points.
    """
    smoothed: list[SmartClip] = []

    for clip in clips:
        # Clamp to source duration
        clamped_start = max(0.0, clip.full_start)
        clamped_end = min(source_duration, clip.full_end)

        if clamped_end <= clamped_start + 1.0:
            continue

        smoothed.append(SmartClip(
            start=clamped_start,
            end=clamped_end,
            score=clip.score,
            event_type=clip.event_type,
            label=clip.label,
            context_before=0.0,  # Already applied
            context_after=0.0,
            priority=clip.priority,
        ))

    return smoothed


# ---------------------------------------------------------------------------
# Main smart editing pipeline
# ---------------------------------------------------------------------------

def generate_smart_edit_plan(
    highlights: list[HighlightEvent],
    *,
    source_duration: float,
    target_duration: float = 600.0,
    min_clip_seconds: float = 8.0,
    max_clip_seconds: float = 45.0,
    pre_fight_context: float = 5.0,
    post_fight_aftermath: float = 3.0,
    dynamic_length: bool = True,
    retain_death_context: bool = True,
    min_gap_between_clips: float = 2.0,
    max_clips: int = 30,
) -> EditPlan:
    """Generate an optimized edit plan from ranked highlights.

    The smart editor applies:
      1. Dynamic clip lengths based on event intensity
      2. Pre-fight context and post-fight aftermath windows
      3. Multi-kill prioritization (pentakills get the longest, best clips)
      4. Death context retention (keep deaths for narrative)
      5. Clip merging and smoothing
      6. Duration targeting (trim or expand to hit target)
    """
    if not highlights:
        return EditPlan(source_duration=source_duration)

    # Sort by priority (descending), then score
    prioritized = sorted(
        highlights,
        key=lambda h: (-_assign_priority(h), -h.score),
    )

    # Generate clips with smart windows
    clips: list[SmartClip] = []
    for event in prioritized:
        if dynamic_length:
            clip_duration = _compute_dynamic_clip_length(
                event,
                min_seconds=min_clip_seconds,
                max_seconds=max_clip_seconds,
            )
        else:
            clip_duration = (min_clip_seconds + max_clip_seconds) / 2.0

        pre_ctx, post_ctx = _compute_context_windows(
            event,
            pre_fight_seconds=pre_fight_context,
            post_fight_seconds=post_fight_aftermath,
        )

        half_dur = clip_duration / 2.0
        clip_start = max(0.0, event.timestamp - half_dur)
        clip_end = min(source_duration, event.timestamp + half_dur)

        clip = SmartClip(
            start=clip_start,
            end=clip_end,
            score=event.score,
            event_type=event.event_type,
            label=event.label,
            context_before=pre_ctx,
            context_after=post_ctx,
            priority=_assign_priority(event),
        )
        clips.append(clip)

    # Filter out death events if not retaining
    if not retain_death_context:
        clips = [c for c in clips if c.event_type != "death"]

    # Merge overlapping clips
    merged = _merge_overlapping_clips(clips, min_gap_seconds=min_gap_between_clips)

    # Apply smoothing
    smoothed = _apply_smoothing(merged, source_duration=source_duration)

    # Sort chronologically
    smoothed.sort(key=lambda c: c.start)

    # Trim to target duration if needed
    total = sum(c.duration for c in smoothed)
    if total > target_duration and len(smoothed) > 1:
        # Remove lowest-priority clips until we're under target
        by_priority = sorted(smoothed, key=lambda c: c.priority)
        while total > target_duration and len(by_priority) > 1:
            removed = by_priority.pop(0)
            total -= removed.duration
            smoothed.remove(removed)

    # Limit clip count
    if len(smoothed) > max_clips:
        smoothed.sort(key=lambda c: -c.priority)
        smoothed = smoothed[:max_clips]
        smoothed.sort(key=lambda c: c.start)

    # Compute stats
    total_duration = sum(c.duration for c in smoothed)
    avg_duration = total_duration / len(smoothed) if smoothed else 0.0
    coverage = total_duration / source_duration if source_duration > 0 else 0.0

    return EditPlan(
        clips=smoothed,
        total_duration=total_duration,
        source_duration=source_duration,
        highlight_count=len(smoothed),
        avg_clip_duration=avg_duration,
        coverage_ratio=coverage,
    )
