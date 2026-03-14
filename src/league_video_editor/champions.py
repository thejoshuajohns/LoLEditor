"""Champion detection from OCR/transcript cues.

Fetches champion metadata from Riot Data Dragon to match champion names
found in video analysis cues.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

from .models import EditorError, TranscriptionCue

# ---------------------------------------------------------------------------
# Data Dragon URLs
# ---------------------------------------------------------------------------

_VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
_CHAMPION_JSON_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"

# Local cache dir for downloaded champion data.
_CACHE_DIR = Path.home() / ".cache" / "lol-video-editor" / "champions"


def _fetch_json(url: str) -> object:
    req = Request(url, headers={"User-Agent": "LoLEditor/1.0"})
    with urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def _get_latest_version() -> str:
    versions = _fetch_json(_VERSIONS_URL)
    if isinstance(versions, list) and versions:
        return str(versions[0])
    raise EditorError("Could not determine latest Data Dragon version.")


def fetch_champion_list() -> dict[str, dict]:
    """Fetch champion data from Data Dragon.

    Returns a dict keyed by champion string ID (e.g. "Aatrox") with fields:
    id, key (numeric), name, tags, etc.

    Results are cached locally for 24 hours.
    """
    cache_file = _CACHE_DIR / "champion_list.json"
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 24:
            return json.loads(cache_file.read_text(encoding="utf-8"))

    version = _get_latest_version()
    url = _CHAMPION_JSON_URL.format(version=version)
    raw = _fetch_json(url)
    if not isinstance(raw, dict) or "data" not in raw:
        raise EditorError("Unexpected champion JSON structure from Data Dragon.")
    data = raw["data"]
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def _build_champion_lookup() -> dict[str, str]:
    """Build a lowercase-name -> champion-id lookup dict."""
    data = fetch_champion_list()
    lookup: dict[str, str] = {}
    for champ_id, info in data.items():
        lookup[champ_id.lower()] = champ_id
        name = info.get("name", "")
        if name:
            lookup[name.lower()] = champ_id
        joined = re.sub(r"\s+", "", name.lower())
        if joined and joined != name.lower():
            lookup[joined] = champ_id
    return lookup


_CHAMPION_LOOKUP: dict[str, str] | None = None


def _get_champion_lookup() -> dict[str, str]:
    global _CHAMPION_LOOKUP
    if _CHAMPION_LOOKUP is None:
        _CHAMPION_LOOKUP = _build_champion_lookup()
    return _CHAMPION_LOOKUP


def detect_champion_from_cues(
    cues: list[TranscriptionCue],
) -> str | None:
    """Identify which champion the player is using from transcript/OCR cues.

    Returns the champion string ID (e.g. "Zed") or None if not detected.
    """
    if not cues:
        return None

    lookup = _get_champion_lookup()
    names_by_length = sorted(lookup.keys(), key=len, reverse=True)
    patterns = [
        (re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE), lookup[name])
        for name in names_by_length
        if len(name) >= 3
    ]

    mentions: Counter[str] = Counter()
    for cue in cues:
        text = cue.text.lower()
        for pattern, champ_id in patterns:
            if pattern.search(text):
                mentions[champ_id] += 1

    if not mentions:
        return None

    top_champ, count = mentions.most_common(1)[0]
    return top_champ if count >= 1 else None


def get_champion_display_name(champion_id: str) -> str:
    """Get the display name (e.g. 'MissFortune' -> 'Miss Fortune')."""
    data = fetch_champion_list()
    info = data.get(champion_id)
    if info is None:
        return champion_id
    return info.get("name", champion_id)


def get_champion_tags(champion_id: str) -> list[str]:
    """Get champion role tags (e.g. ['Assassin', 'Fighter'])."""
    data = fetch_champion_list()
    info = data.get(champion_id)
    if info is None:
        return []
    return list(info.get("tags", []))
