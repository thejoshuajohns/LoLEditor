"""Persistent configuration system.

Supports two levels of configuration:
  1. GlobalConfig  — user-wide defaults (~/.config/lol-video-editor/config.json)
  2. ProjectConfig — per-video project settings (saved alongside edit plan)

Values cascade: CLI flags > project config > global config > built-in defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


_GLOBAL_CONFIG_PATH = Path.home() / ".config" / "lol-video-editor" / "config.json"


@dataclass
class ChampionContext:
    """Metadata about the player's champion for overlays and branding."""

    champion_name: str = ""
    champion_png: str = ""  # Path to champion portrait PNG
    lane: str = ""  # top, jungle, mid, bot, support
    role: str = ""  # carry, tank, assassin, mage, support, etc.
    player_name: str = ""
    rank: str = ""  # iron, bronze, silver, gold, platinum, emerald, diamond, master, grandmaster, challenger
    patch: str = ""  # e.g. "14.10"
    kills: int | None = None
    deaths: int | None = None
    assists: int | None = None

    def is_set(self) -> bool:
        return bool(self.champion_name.strip())


@dataclass
class OverlayConfig:
    """Settings for auto-generated video overlays."""

    enable_intro: bool = True
    enable_outro: bool = True
    enable_kda_overlay: bool = True
    enable_champion_portrait: bool = True
    enable_transitions: bool = True
    enable_subscribe_animation: bool = False
    intro_duration_seconds: float = 4.0
    outro_duration_seconds: float = 6.0
    transition_style: str = "crossfade"  # crossfade, wipe, fade_black, glitch
    transition_duration_seconds: float = 0.8
    kda_position: str = "top-right"  # top-left, top-right, bottom-left, bottom-right
    portrait_position: str = "bottom-left"
    font_family: str = "auto"


@dataclass
class DetectionConfig:
    """Advanced highlight detection settings."""

    # Scene detection
    scene_threshold: float = 0.35
    scene_weight: float = 0.15

    # Kill feed detection (OCR on kill feed region)
    killfeed_enabled: bool = True
    killfeed_weight: float = 0.25
    killfeed_region: str = "auto"  # auto-detect or "x,y,w,h"

    # Combat intensity (motion + saturation bursts)
    combat_intensity_enabled: bool = True
    combat_intensity_weight: float = 0.20

    # Objective detection
    objective_detection_enabled: bool = True
    objective_weight: float = 0.15

    # Scoreboard spike detection
    scoreboard_detection_enabled: bool = True
    scoreboard_weight: float = 0.10

    # Whisper transcript
    whisper_enabled: bool = True
    whisper_weight: float = 0.10

    # Audio excitement (volume spikes, vocal energy)
    audio_excitement_enabled: bool = True
    audio_excitement_weight: float = 0.15

    # Multi-kill prioritization
    multi_kill_bonus: float = 0.30
    pentakill_bonus: float = 0.50


@dataclass
class EditingConfig:
    """Smart editing behavior settings."""

    # Context windows
    pre_fight_context_seconds: float = 5.0
    post_fight_aftermath_seconds: float = 3.0

    # Dynamic clip length
    min_clip_seconds: float = 8.0
    max_clip_seconds: float = 45.0
    dynamic_length_enabled: bool = True

    # Smoothing
    smoothing_enabled: bool = True
    min_gap_between_clips: float = 2.0

    # Death context
    retain_death_context: bool = True
    death_context_before: float = 8.0
    death_context_after: float = 3.0

    # Target output
    target_duration_seconds: float = 600.0
    max_clips: int = 30


@dataclass
class UploadConfig:
    """YouTube upload settings."""

    auto_upload: bool = False
    privacy: str = "private"  # private, unlisted, public
    category_id: str = "20"  # Gaming
    channel_name: str = ""
    default_tags: list[str] = field(
        default_factory=lambda: ["leagueoflegends", "lolhighlights", "gaming"]
    )


@dataclass
class OutputConfig:
    """Output format settings."""

    format: str = "youtube"  # youtube, tiktok, twitch_clip, shorts
    resolution: str = "1080p"  # 720p, 1080p, 1440p, 4k
    framerate: int = 60
    codec: str = "auto"  # auto, libx264, h264_videotoolbox, h264_nvenc
    crf: int = 20
    preset: str = "fast"
    profile: str = "balanced"  # fast, balanced, quality

    @property
    def width(self) -> int:
        return {
            "720p": 1280, "1080p": 1920, "1440p": 2560, "4k": 3840,
        }.get(self.resolution, 1920)

    @property
    def height(self) -> int:
        return {
            "720p": 720, "1080p": 1080, "1440p": 1440, "4k": 2160,
        }.get(self.resolution, 1080)


@dataclass
class ContentConfig:
    """Generated publishing metadata and branding copy."""

    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    thumbnail_headline: str = ""
    auto_generate_title: bool = True
    auto_generate_description: bool = True
    auto_generate_tags: bool = True
    auto_generate_chapters: bool = True


@dataclass
class ProjectConfig:
    """Full configuration for a single video project."""

    # Input
    input_path: str = ""
    output_dir: str = ""

    # Champion context
    champion: ChampionContext = field(default_factory=ChampionContext)

    # Detection
    detection: DetectionConfig = field(default_factory=DetectionConfig)

    # Editing
    editing: EditingConfig = field(default_factory=EditingConfig)

    # Overlays
    overlays: OverlayConfig = field(default_factory=OverlayConfig)

    # Output
    output: OutputConfig = field(default_factory=OutputConfig)

    # Content metadata
    content: ContentConfig = field(default_factory=ContentConfig)

    # Upload
    upload: UploadConfig = field(default_factory=UploadConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "ProjectConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        return _dict_to_project_config(data)


@dataclass
class GlobalConfig:
    """User-wide default settings."""

    default_profile: str = "balanced"
    default_output_dir: str = ""
    cache_dir: str = str(Path.home() / ".cache" / "lol-video-editor")
    cache_max_gb: float = 5.0
    cache_ttl_days: float = 30.0

    # Default champion context (reused across projects)
    player_name: str = ""
    channel_name: str = ""

    # Detection defaults
    detection: DetectionConfig = field(default_factory=DetectionConfig)

    # Editing defaults
    editing: EditingConfig = field(default_factory=EditingConfig)

    # Overlay defaults
    overlays: OverlayConfig = field(default_factory=OverlayConfig)

    # Output defaults
    output: OutputConfig = field(default_factory=OutputConfig)

    # Content defaults
    content: ContentConfig = field(default_factory=ContentConfig)

    # Upload defaults
    upload: UploadConfig = field(default_factory=UploadConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: Path | None = None) -> None:
        target = path or _GLOBAL_CONFIG_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )


def _nested_dataclass_from_dict(cls, data: dict) -> Any:
    """Recursively construct a dataclass from a dict, ignoring unknown keys."""
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return data

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    field_names = set(field_types.keys())
    filtered = {k: v for k, v in data.items() if k in field_names}

    # Resolve nested dataclasses
    type_map = {
        "ChampionContext": ChampionContext,
        "OverlayConfig": OverlayConfig,
        "DetectionConfig": DetectionConfig,
        "EditingConfig": EditingConfig,
        "UploadConfig": UploadConfig,
        "OutputConfig": OutputConfig,
        "ContentConfig": ContentConfig,
    }
    for key, val in filtered.items():
        if isinstance(val, dict):
            type_name = field_types.get(key, "")
            # Strip quotes and Optional wrapper
            clean_name = str(type_name).replace("'", "").strip()
            if clean_name in type_map:
                filtered[key] = _nested_dataclass_from_dict(type_map[clean_name], val)

    return cls(**filtered)


def _dict_to_project_config(data: dict) -> ProjectConfig:
    return _nested_dataclass_from_dict(ProjectConfig, data)


def load_global_config(path: Path | None = None) -> GlobalConfig:
    """Load global config from disk, or return defaults if not found."""
    target = path or _GLOBAL_CONFIG_PATH
    if not target.exists():
        return GlobalConfig()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return _nested_dataclass_from_dict(GlobalConfig, data)
    except (json.JSONDecodeError, OSError, TypeError):
        return GlobalConfig()


def load_project_config(path: Path) -> ProjectConfig:
    """Load project config from disk, or return defaults if not found."""
    if not path.exists():
        return ProjectConfig()
    try:
        return ProjectConfig.load(path)
    except (json.JSONDecodeError, OSError, TypeError):
        return ProjectConfig()
