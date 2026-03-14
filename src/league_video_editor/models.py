"""Shared dataclasses, constants, and compiled patterns.

Keeping these here avoids circular imports between the analysis, rendering,
and CLI modules.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class EditorError(RuntimeError):
    """Raised when the editor cannot proceed with the current configuration."""


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


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

    @property
    def center(self) -> float:
        return (self.start + self.end) / 2.0


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


# ---------------------------------------------------------------------------
# Regex patterns (compiled once at import time)
# ---------------------------------------------------------------------------

SCENE_PTS_PATTERN = re.compile(r"pts_time:(-?\d+(?:\.\d+)?)")
FFMPEG_TIME_PATTERN = re.compile(r"time=(\d{2}:\d{2}:\d{2}(?:\.\d+)?)")
SIGNALSTATS_FRAME_PATTERN = re.compile(
    r"frame:\s*\d+\s+pts:\s*-?\d+\s+pts_time:(-?\d+(?:\.\d+)?)"
)
SIGNALSTATS_YDIF_PATTERN = re.compile(
    r"lavfi\.signalstats\.YDIF=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
SIGNALSTATS_SATAVG_PATTERN = re.compile(
    r"lavfi\.signalstats\.SATAVG=([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
)
CROPDETECT_PATTERN = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")
WHISPER_TIMESTAMP_PATTERN = re.compile(
    r"^(\d{1,2}):(\d{2}):(\d{2})(?:[.,](\d{1,3}))?$"
)
LOUDNORM_NONFINITE_PATTERN = re.compile(
    r"input contains.*nan.*inf", re.IGNORECASE | re.DOTALL
)

# ---------------------------------------------------------------------------
# Loudnorm
# ---------------------------------------------------------------------------

LOUDNORM_FILTER = "loudnorm=I=-14:LRA=11:TP=-1.5"
LOUDNORM_ANALYSIS_FILTER = f"{LOUDNORM_FILTER}:print_format=json"

# ---------------------------------------------------------------------------
# Encoder constants
# ---------------------------------------------------------------------------

VIDEO_ENCODERS = ("libx264", "h264_videotoolbox", "h264_nvenc", "h264_qsv", "h264_amf")

# ---------------------------------------------------------------------------
# Analysis defaults
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Thumbnail defaults
# ---------------------------------------------------------------------------

DEFAULT_THUMBNAIL_WIDTH = 1280
DEFAULT_THUMBNAIL_HEIGHT = 720
DEFAULT_THUMBNAIL_QUALITY = 2
DEFAULT_THUMBNAIL_SCENE_THRESHOLD = 0.20
DEFAULT_WATCHABILITY_SCENE_THRESHOLD = 0.35
DEFAULT_THUMBNAIL_VISION_SAMPLE_FPS = 0.75
DEFAULT_THUMBNAIL_VISION_WINDOW_SECONDS = 8.0
DEFAULT_THUMBNAIL_VISION_STEP_SECONDS = 4.0
DEFAULT_THUMBNAIL_CHAMPION_SCALE = 0.55
DEFAULT_THUMBNAIL_CHAMPION_ANCHOR = "right"
DEFAULT_THUMBNAIL_HEADLINE_SIZE = 118
DEFAULT_THUMBNAIL_HEADLINE_COLOR = "white"
DEFAULT_THUMBNAIL_HEADLINE_Y_RATIO = 0.06

# ---------------------------------------------------------------------------
# Description defaults
# ---------------------------------------------------------------------------

DEFAULT_DESCRIPTION_OUTPUT = "youtube-description.txt"
DEFAULT_DESCRIPTION_TITLE_COUNT = 3
DEFAULT_DESCRIPTION_MAX_MOMENTS = 8

# ---------------------------------------------------------------------------
# Adaptive thresholds
# ---------------------------------------------------------------------------

ADAPTIVE_AI_CUE_THRESHOLD_MIN = 0.12
ADAPTIVE_AI_CUE_THRESHOLD_FACTORS = (1.0, 0.82, 0.65, 0.48)
ADAPTIVE_SCENE_THRESHOLD_MIN = 0.08
ADAPTIVE_SCENE_THRESHOLD_FACTORS = (1.0, 0.75, 0.6, 0.45)

# ---------------------------------------------------------------------------
# Event context windows
# ---------------------------------------------------------------------------

EVENT_CONTEXT_DEATH_BEFORE_SECONDS = 18.0
EVENT_CONTEXT_DEATH_AFTER_SECONDS = 9.0
EVENT_CONTEXT_COMBAT_BEFORE_SECONDS = 12.0
EVENT_CONTEXT_COMBAT_AFTER_SECONDS = 8.0
EVENT_CONTEXT_AI_BEFORE_SECONDS = 14.0
EVENT_CONTEXT_AI_AFTER_SECONDS = 9.0
AI_WINDOW_BOOST_RADIUS_SECONDS = 22.0

# ---------------------------------------------------------------------------
# Encoder profiles
# ---------------------------------------------------------------------------
#
# A profile sets default values for encoder choice, CRF/bitrate strategy,
# preset, two-pass loudnorm, and analysis sample FPS.  Individual CLI flags
# always override profile defaults.
#

ENCODER_PROFILES: dict[str, dict[str, object]] = {
    "fast": {
        # Prefer VideoToolbox on Apple Silicon; fall back to libx264 fast.
        "video_encoder": "auto",
        "crf": 23,
        "preset": "fast",
        "two_pass_loudnorm": False,
        # Reduce vision sampling for faster analysis on first-pass iteration.
        "vision_sample_fps": 0.5,
    },
    "balanced": {
        # VideoToolbox when available, else libx264.
        "video_encoder": "auto",
        "crf": 20,
        "preset": "fast",
        "two_pass_loudnorm": False,
        "vision_sample_fps": 1.0,
    },
    "quality": {
        # Always libx264 for best compatibility + quality.
        "video_encoder": "libx264",
        "crf": 18,
        "preset": "slow",
        "two_pass_loudnorm": True,
        "vision_sample_fps": 1.0,
    },
}

DEFAULT_PROFILE = "balanced"

# ---------------------------------------------------------------------------
# Hype phrase weights used by transcript scorer
# ---------------------------------------------------------------------------

TRANSCRIPT_HYPE_WEIGHTS: dict[str, float] = {
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


TRANSCRIPT_HYPE_PATTERNS: tuple[tuple[str, float, re.Pattern[str]], ...] = tuple(
    (phrase, weight, _compile_hype_phrase_pattern(phrase))
    for phrase, weight in TRANSCRIPT_HYPE_WEIGHTS.items()
)
