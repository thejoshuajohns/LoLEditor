"""Advanced multi-signal highlight detection engine.

Combines multiple detection signals with weighted scoring to identify
the most important moments in a League of Legends recording.

Signals:
  1. Scene changes — visual cuts indicate action transitions
  2. Kill feed — OCR on the kill feed region to detect kills/assists
  3. Combat intensity — motion + saturation burst detection
  4. Objective events — baron, dragon, tower via transcript/OCR keywords
  5. Whisper transcript — caster/player callouts scored by hype weight
  6. Audio excitement — volume RMS spikes indicating hype moments
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from ..models import Segment, TranscriptionCue, VisionWindow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Highlight event types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HighlightEvent:
    """A single detected highlight moment with multi-signal scoring."""

    timestamp: float  # Center timestamp in seconds
    duration: float  # Estimated event duration
    score: float  # Composite weighted score (0.0 - 1.0)
    event_type: str  # kill, multi_kill, objective, teamfight, death, outplay
    label: str  # Human-readable label
    signals: dict[str, float] = field(default_factory=dict)  # Per-signal scores
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def start(self) -> float:
        return max(0.0, self.timestamp - self.duration / 2)

    @property
    def end(self) -> float:
        return self.timestamp + self.duration / 2


# ---------------------------------------------------------------------------
# Signal weights configuration
# ---------------------------------------------------------------------------

@dataclass
class SignalWeights:
    """Configurable weights for each detection signal."""

    scene_change: float = 0.15
    kill_feed: float = 0.25
    combat_intensity: float = 0.20
    objectives: float = 0.15
    scoreboard_spike: float = 0.10
    transcript: float = 0.10
    audio_excitement: float = 0.15

    def normalized(self) -> "SignalWeights":
        total = (
            self.scene_change + self.kill_feed + self.combat_intensity
            + self.objectives + self.scoreboard_spike + self.transcript + self.audio_excitement
        )
        if total <= 0:
            return SignalWeights()
        factor = 1.0 / total
        return SignalWeights(
            scene_change=self.scene_change * factor,
            kill_feed=self.kill_feed * factor,
            combat_intensity=self.combat_intensity * factor,
            objectives=self.objectives * factor,
            scoreboard_spike=self.scoreboard_spike * factor,
            transcript=self.transcript * factor,
            audio_excitement=self.audio_excitement * factor,
        )


# ---------------------------------------------------------------------------
# Event type classification
# ---------------------------------------------------------------------------

# Keywords that indicate specific event types
MULTI_KILL_KEYWORDS = frozenset({
    "pentakill", "quadra kill", "triple kill", "double kill",
})
OBJECTIVE_KEYWORDS = frozenset({
    "baron", "baron nashor", "nashor", "dragon", "elder dragon",
    "rift herald", "turret", "tower", "inhibitor", "nexus",
})
KILL_KEYWORDS = frozenset({
    "kill", "slain", "enemy slain", "first blood", "shutdown",
    "killing spree", "rampage", "unstoppable", "godlike", "legendary",
})
TEAMFIGHT_KEYWORDS = frozenset({
    "teamfight", "fight", "ace",
})

# Event type priority bonuses
EVENT_TYPE_BONUSES: dict[str, float] = {
    "pentakill": 0.50,
    "quadra_kill": 0.40,
    "triple_kill": 0.30,
    "double_kill": 0.15,
    "baron_steal": 0.45,
    "ace": 0.35,
    "first_blood": 0.20,
    "shutdown": 0.25,
    "elder_dragon": 0.35,
    "baron": 0.30,
    "nexus": 0.40,
}


# ---------------------------------------------------------------------------
# Kill feed detection
# ---------------------------------------------------------------------------

KILL_FEED_PATTERNS: list[tuple[str, str, float]] = [
    # (pattern_description, regex-friendly keyword, score)
    ("pentakill", "pentakill", 1.00),
    ("quadra kill", "quadra", 0.90),
    ("triple kill", "triple", 0.82),
    ("double kill", "double kill", 0.66),
    ("has slain", "has slain", 0.55),
    ("killed", "killed", 0.50),
    ("executed", "executed", 0.40),
    ("shutdown", "shutdown", 0.75),
    ("ace", "ace", 0.80),
]


def detect_kill_feed_events(
    ocr_cues: list[TranscriptionCue],
    *,
    kill_feed_region: str = "auto",
) -> list[HighlightEvent]:
    """Extract kill events from OCR cues detected in the kill feed region.

    The kill feed in League of Legends is in the top-right corner. OCR cues
    from that region are scored for kill-related keywords.
    """
    events: list[HighlightEvent] = []

    for cue in ocr_cues:
        text_lower = cue.text.lower()
        best_score = 0.0
        best_label = ""

        for desc, keyword, score in KILL_FEED_PATTERNS:
            if keyword in text_lower:
                if score > best_score:
                    best_score = score
                    best_label = desc

        if best_score > 0.0:
            event_type = "multi_kill" if best_score >= 0.66 else "kill"
            events.append(HighlightEvent(
                timestamp=cue.center,
                duration=cue.end - cue.start,
                score=best_score,
                event_type=event_type,
                label=best_label.title(),
                signals={"kill_feed": best_score},
                metadata={"source_text": cue.text},
            ))

    return events


# ---------------------------------------------------------------------------
# Combat intensity detection
# ---------------------------------------------------------------------------

def detect_combat_intensity(
    vision_windows: list[VisionWindow],
    *,
    intensity_threshold: float = 0.60,
    burst_threshold: float = 0.75,
) -> list[HighlightEvent]:
    """Detect high-intensity combat moments from vision analysis.

    Looks for sustained high motion + saturation (teamfights) and
    sudden bursts (assassinations, burst combos).
    """
    if not vision_windows:
        return []

    events: list[HighlightEvent] = []
    motions = [w.motion for w in vision_windows if math.isfinite(w.motion)]
    sats = [w.saturation for w in vision_windows if math.isfinite(w.saturation)]

    if not motions or not sats:
        return []

    motion_p75 = sorted(motions)[int(len(motions) * 0.75)]
    motion_p90 = sorted(motions)[int(len(motions) * 0.90)]
    sat_mean = sum(sats) / len(sats)

    for w in vision_windows:
        if not math.isfinite(w.motion) or not math.isfinite(w.saturation):
            continue

        # Combat intensity combines motion, saturation, and scene density
        motion_norm = min(1.0, w.motion / max(1.0, motion_p90)) if motion_p90 > 0 else 0.0
        sat_norm = min(1.0, w.saturation / max(1.0, sat_mean * 1.5)) if sat_mean > 0 else 0.0
        density_norm = min(1.0, w.scene_density)

        intensity = (
            0.50 * motion_norm
            + 0.30 * sat_norm
            + 0.20 * density_norm
        )

        if intensity >= intensity_threshold:
            is_burst = w.motion >= motion_p75 and intensity >= burst_threshold
            event_type = "burst_combo" if is_burst else "teamfight"
            label = "Burst Combo" if is_burst else "Intense Combat"

            events.append(HighlightEvent(
                timestamp=w.center,
                duration=w.end - w.start,
                score=intensity,
                event_type=event_type,
                label=label,
                signals={
                    "combat_intensity": intensity,
                    "motion": motion_norm,
                    "saturation": sat_norm,
                    "scene_density": density_norm,
                },
            ))

    return events


# ---------------------------------------------------------------------------
# Objective event detection
# ---------------------------------------------------------------------------

OBJECTIVE_CUE_SCORES: dict[str, float] = {
    "baron nashor": 0.90,
    "baron": 0.85,
    "nashor": 0.85,
    "elder dragon": 0.88,
    "dragon": 0.60,
    "rift herald": 0.55,
    "nexus": 0.95,
    "inhibitor": 0.50,
    "turret": 0.40,
    "tower": 0.40,
}

SCOREBOARD_PATTERN = re.compile(r"\b(\d{1,2})\s*[-:]\s*(\d{1,2})\b")


def detect_objective_events(
    cues: list[TranscriptionCue],
) -> list[HighlightEvent]:
    """Detect objective-related events from transcript/OCR cues."""
    events: list[HighlightEvent] = []

    for cue in cues:
        kw_set = set(cue.keywords)
        best_score = 0.0
        best_label = ""

        for keyword, score in OBJECTIVE_CUE_SCORES.items():
            if keyword in kw_set:
                if score > best_score:
                    best_score = score
                    best_label = keyword.title()

        if best_score > 0.0:
            events.append(HighlightEvent(
                timestamp=cue.center,
                duration=cue.end - cue.start,
                score=best_score,
                event_type="objective",
                label=best_label,
                signals={"objectives": best_score},
                metadata={"keywords": list(cue.keywords)},
            ))

    return events


def detect_scoreboard_spikes(
    cues: list[TranscriptionCue],
    *,
    minimum_delta: int = 2,
) -> list[HighlightEvent]:
    """Detect score swings from OCR/transcript scoreboard cues."""

    parsed: list[tuple[float, int, int, str]] = []
    for cue in sorted(cues, key=lambda item: item.start):
        match = SCOREBOARD_PATTERN.search(cue.text)
        if not match:
            continue
        left = int(match.group(1))
        right = int(match.group(2))
        total = left + right
        lead = abs(left - right)
        if total <= 0 or total > 90:
            continue
        parsed.append((cue.center, total, lead, cue.text))

    if len(parsed) < 2:
        return []

    events: list[HighlightEvent] = []
    prev_time, prev_total, prev_lead, _ = parsed[0]
    for center, total, lead, raw_text in parsed[1:]:
        total_delta = total - prev_total
        lead_delta = abs(lead - prev_lead)
        if total_delta < minimum_delta and lead_delta < minimum_delta:
            prev_time, prev_total, prev_lead = center, total, lead
            continue
        score = min(1.0, 0.25 + 0.12 * max(0, total_delta) + 0.08 * lead_delta)
        events.append(
            HighlightEvent(
                timestamp=center,
                duration=max(4.0, center - prev_time),
                score=score,
                event_type="scoreboard_spike",
                label="Scoreboard Swing",
                signals={"scoreboard_spike": score},
                metadata={
                    "source_text": raw_text,
                    "total_delta": total_delta,
                    "lead_delta": lead_delta,
                },
            )
        )
        prev_time, prev_total, prev_lead = center, total, lead

    return events


# ---------------------------------------------------------------------------
# Audio excitement detection
# ---------------------------------------------------------------------------

def detect_audio_excitement(
    audio_rms_samples: list[tuple[float, float]],
    *,
    excitement_threshold: float = 0.70,
    window_seconds: float = 3.0,
) -> list[HighlightEvent]:
    """Detect excitement spikes from audio RMS levels.

    Parameters
    ----------
    audio_rms_samples:
        List of (timestamp, rms_level) tuples from audio analysis.
    excitement_threshold:
        Normalized threshold (0-1) above which a spike is considered exciting.
    window_seconds:
        How long an excitement spike must be sustained.
    """
    if not audio_rms_samples:
        return []

    rms_values = [r for _, r in audio_rms_samples]
    if not rms_values:
        return []

    rms_max = max(rms_values)
    rms_mean = sum(rms_values) / len(rms_values)

    if rms_max <= 0:
        return []

    events: list[HighlightEvent] = []
    in_spike = False
    spike_start = 0.0
    spike_peak = 0.0

    for ts, rms in audio_rms_samples:
        normalized = (rms - rms_mean) / max(0.01, rms_max - rms_mean)

        if normalized >= excitement_threshold:
            if not in_spike:
                in_spike = True
                spike_start = ts
                spike_peak = normalized
            else:
                spike_peak = max(spike_peak, normalized)
        else:
            if in_spike:
                spike_duration = ts - spike_start
                if spike_duration >= window_seconds:
                    events.append(HighlightEvent(
                        timestamp=(spike_start + ts) / 2,
                        duration=spike_duration,
                        score=min(1.0, spike_peak),
                        event_type="audio_excitement",
                        label="Audio Spike",
                        signals={"audio_excitement": spike_peak},
                    ))
                in_spike = False

    return events


# ---------------------------------------------------------------------------
# Composite highlight scoring engine
# ---------------------------------------------------------------------------

def merge_nearby_events(
    events: list[HighlightEvent],
    *,
    merge_window_seconds: float = 8.0,
) -> list[HighlightEvent]:
    """Merge events that are close together into single highlights."""
    if not events:
        return []

    sorted_events = sorted(events, key=lambda e: e.timestamp)
    merged: list[HighlightEvent] = []
    current_group: list[HighlightEvent] = [sorted_events[0]]

    for event in sorted_events[1:]:
        group_end = max(e.end for e in current_group)
        if event.start <= group_end + merge_window_seconds:
            current_group.append(event)
        else:
            merged.append(_merge_event_group(current_group))
            current_group = [event]

    if current_group:
        merged.append(_merge_event_group(current_group))

    return merged


def _merge_event_group(group: list[HighlightEvent]) -> HighlightEvent:
    """Merge a group of overlapping events into one."""
    if len(group) == 1:
        return group[0]

    # Take the highest-scoring event as the base
    best = max(group, key=lambda e: e.score)
    start = min(e.start for e in group)
    end = max(e.end for e in group)
    center = (start + end) / 2.0

    # Combine signals from all events
    combined_signals: dict[str, float] = {}
    for event in group:
        for signal, value in event.signals.items():
            combined_signals[signal] = max(combined_signals.get(signal, 0.0), value)

    # Boost score for multi-signal confirmation
    signal_count = len(combined_signals)
    confirmation_bonus = min(0.20, signal_count * 0.05)

    # Combine metadata
    all_types = {e.event_type for e in group}
    combined_metadata: dict[str, object] = {
        "merged_event_count": len(group),
        "event_types": sorted(all_types),
    }

    # Pick the most specific label
    label_priority = [
        "pentakill", "quadra_kill", "triple_kill", "baron_steal",
        "ace", "nexus", "elder_dragon", "baron", "shutdown",
    ]
    label = best.label
    for event in group:
        for priority_type in label_priority:
            if event.event_type == priority_type:
                label = event.label
                break

    return HighlightEvent(
        timestamp=center,
        duration=end - start,
        score=min(1.0, best.score + confirmation_bonus),
        event_type=best.event_type,
        label=label,
        signals=combined_signals,
        metadata=combined_metadata,
    )


def rank_highlights(
    events: list[HighlightEvent],
    weights: SignalWeights | None = None,
    *,
    max_highlights: int = 30,
    min_score: float = 0.25,
) -> list[HighlightEvent]:
    """Score and rank highlight events using weighted multi-signal scoring.

    Returns the top N highlights sorted by composite score (descending).
    """
    if not events:
        return []

    w = (weights or SignalWeights()).normalized()

    scored: list[tuple[float, HighlightEvent]] = []
    for event in events:
        signals = event.signals

        composite = (
            w.scene_change * signals.get("scene_change", 0.0)
            + w.kill_feed * signals.get("kill_feed", 0.0)
            + w.combat_intensity * signals.get("combat_intensity", 0.0)
            + w.objectives * signals.get("objectives", 0.0)
            + w.scoreboard_spike * signals.get("scoreboard_spike", 0.0)
            + w.transcript * signals.get("transcript", 0.0)
            + w.audio_excitement * signals.get("audio_excitement", 0.0)
        )

        # Apply event type bonuses
        for bonus_type, bonus_value in EVENT_TYPE_BONUSES.items():
            if event.event_type == bonus_type or bonus_type in event.label.lower().replace(" ", "_"):
                composite = min(1.0, composite + bonus_value)
                break

        # Multi-signal confirmation bonus
        active_signals = sum(1 for v in signals.values() if v > 0.1)
        if active_signals >= 3:
            composite = min(1.0, composite * 1.15)
        elif active_signals >= 2:
            composite = min(1.0, composite * 1.08)

        if composite >= min_score:
            scored.append((composite, event))

    scored.sort(key=lambda x: (-x[0], x[1].timestamp))

    return [
        HighlightEvent(
            timestamp=event.timestamp,
            duration=event.duration,
            score=score,
            event_type=event.event_type,
            label=event.label,
            signals=event.signals,
            metadata=event.metadata,
        )
        for score, event in scored[:max_highlights]
    ]


# ---------------------------------------------------------------------------
# High-level detection pipeline
# ---------------------------------------------------------------------------

def run_highlight_detection(
    *,
    scene_events: list[float],
    vision_windows: list[VisionWindow],
    whisper_cues: list[TranscriptionCue],
    ocr_cues: list[TranscriptionCue],
    audio_rms_samples: list[tuple[float, float]] | None = None,
    weights: SignalWeights | None = None,
    max_highlights: int = 30,
    duration_seconds: float = 0.0,
) -> list[HighlightEvent]:
    """Run the full multi-signal highlight detection pipeline.

    This is the main entry point that combines all detection signals.
    """
    all_events: list[HighlightEvent] = []

    # 1. Scene change events
    for ts in scene_events:
        # Find the nearest vision window to get context
        nearest_window = _find_nearest_window(ts, vision_windows)
        scene_score = 0.3  # Base score for scene events
        if nearest_window and nearest_window.score > 0.5:
            scene_score = min(1.0, 0.3 + nearest_window.score * 0.4)
        all_events.append(HighlightEvent(
            timestamp=ts,
            duration=2.0,
            score=scene_score,
            event_type="scene_change",
            label="Scene Change",
            signals={"scene_change": scene_score},
        ))

    # 2. Kill feed detection from OCR
    kill_events = detect_kill_feed_events(ocr_cues)
    all_events.extend(kill_events)
    logger.info("Detected %d kill feed events", len(kill_events))

    # 3. Combat intensity from vision
    combat_events = detect_combat_intensity(vision_windows)
    all_events.extend(combat_events)
    logger.info("Detected %d combat intensity events", len(combat_events))

    # 4. Objective events from transcripts
    objective_events = detect_objective_events(whisper_cues + ocr_cues)
    all_events.extend(objective_events)
    logger.info("Detected %d objective events", len(objective_events))

    # 5. Scoreboard spikes from OCR/transcript cues
    scoreboard_events = detect_scoreboard_spikes(whisper_cues + ocr_cues)
    all_events.extend(scoreboard_events)
    logger.info("Detected %d scoreboard spike events", len(scoreboard_events))

    # 6. Transcript hype events
    for cue in whisper_cues:
        if cue.score >= 0.3:
            all_events.append(HighlightEvent(
                timestamp=cue.center,
                duration=cue.end - cue.start,
                score=cue.score,
                event_type="transcript_hype",
                label="Caster Callout",
                signals={"transcript": cue.score},
                metadata={"text": cue.text, "keywords": list(cue.keywords)},
            ))

    # 7. Audio excitement
    if audio_rms_samples:
        audio_events = detect_audio_excitement(audio_rms_samples)
        all_events.extend(audio_events)
        logger.info("Detected %d audio excitement events", len(audio_events))

    # Merge nearby events and rank
    merged = merge_nearby_events(all_events)
    ranked = rank_highlights(merged, weights=weights, max_highlights=max_highlights)

    logger.info(
        "Highlight detection complete: %d raw events -> %d merged -> %d ranked",
        len(all_events), len(merged), len(ranked),
    )

    return ranked


def _find_nearest_window(
    timestamp: float,
    windows: list[VisionWindow],
) -> VisionWindow | None:
    """Find the vision window closest to a timestamp."""
    if not windows:
        return None
    best = None
    best_dist = float("inf")
    for w in windows:
        dist = abs(w.center - timestamp)
        if dist < best_dist:
            best_dist = dist
            best = w
    return best
