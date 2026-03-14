"""Persistent disk cache for expensive analysis steps.

Cache key includes: input file path, size, mtime, and a settings hash so any
change in parameters (scene threshold, sample FPS, etc.) produces a new entry.

Storage layout:
  <cache_dir>/<key[:2]>/<key>_<artifact>.json.gz

Each entry is a gzip-compressed JSON blob.  Size is bounded by ``max_gb`` and
entries are expired by ``ttl_days``.  When the budget is exceeded the oldest
entries are removed first (LRU-by-mtime).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------
# Public defaults (can be overridden via CLI flags)
# ------------------------------------------------------------------

DEFAULT_CACHE_DIR: Path = Path.home() / ".cache" / "lol-video-editor"
DEFAULT_CACHE_MAX_GB: float = 5.0
DEFAULT_CACHE_TTL_DAYS: float = 30.0


# ------------------------------------------------------------------
# Cache key helpers
# ------------------------------------------------------------------


def _file_fingerprint(path: Path) -> str:
    """Return a stable string that changes when the file changes."""
    try:
        stat = path.stat()
        return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime}"
    except OSError:
        return str(path.resolve())


def make_file_cache_key(
    input_path: Path,
    artifact: str,
    *,
    settings: dict[str, object] | None = None,
) -> str:
    """Return a stable hex SHA-1 key for an input file + artifact + settings.

    Parameters
    ----------
    input_path:
        The source video file being analysed.
    artifact:
        Short label identifying what is cached (e.g. ``"scene_events"``).
    settings:
        Dict of analysis parameters that affect the result.  Any change
        (e.g. scene threshold) produces a different key.
    """
    parts = [artifact, _file_fingerprint(input_path)]
    if settings:
        parts.append(json.dumps(settings, sort_keys=True, separators=(",", ":")))
    raw = "|".join(parts)
    return hashlib.sha1(raw.encode()).hexdigest()


# ------------------------------------------------------------------
# Entry dataclass
# ------------------------------------------------------------------


@dataclass
class CacheEntry:
    key: str
    artifact: str
    path: Path
    size_bytes: int
    mtime: float


# ------------------------------------------------------------------
# CacheStore
# ------------------------------------------------------------------


class CacheStore:
    """Disk-backed JSON cache with TTL and GB-budget enforcement.

    Parameters
    ----------
    cache_dir:
        Root directory for all cache files.  Created automatically.
    max_gb:
        Maximum total disk usage.  0 or negative means unlimited.
    ttl_days:
        Entries older than this are considered stale.  0 or negative
        disables TTL (entries live until budget eviction).
    """

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        max_gb: float = DEFAULT_CACHE_MAX_GB,
        ttl_days: float = DEFAULT_CACHE_TTL_DAYS,
    ) -> None:
        self.cache_dir = cache_dir
        self.max_bytes = int(max_gb * 1024**3) if max_gb > 0 else 0
        self.ttl_seconds = ttl_days * 86400 if ttl_days > 0 else 0.0
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _entry_path(self, key: str, artifact: str) -> Path:
        shard = key[:2]
        return self.cache_dir / shard / f"{key}_{artifact}.json.gz"

    def _is_stale(self, path: Path) -> bool:
        if self.ttl_seconds <= 0:
            return False
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            return True
        return age > self.ttl_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str, artifact: str) -> Any | None:
        """Return cached data, or ``None`` if absent/stale/corrupt."""
        path = self._entry_path(key, artifact)
        if not path.exists():
            return None
        if self._is_stale(path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, gzip.BadGzipFile, json.JSONDecodeError, EOFError, ValueError):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def put(self, key: str, artifact: str, data: Any) -> None:
        """Write ``data`` to cache.  Enforces budget after writing."""
        path = self._entry_path(key, artifact)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as fh:
                json.dump(data, fh, separators=(",", ":"))
        except OSError:
            return
        self._enforce_budget()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_entries(self) -> list[CacheEntry]:
        """Return all cache entries sorted newest-first."""
        entries: list[CacheEntry] = []
        try:
            for gz_path in self.cache_dir.rglob("*.json.gz"):
                try:
                    stat = gz_path.stat()
                    stem = gz_path.stem  # "<key>_<artifact>"
                    idx = stem.find("_")
                    if idx < 0:
                        continue
                    cache_key = stem[:idx]
                    artifact = stem[idx + 1 :]
                    entries.append(
                        CacheEntry(
                            key=cache_key,
                            artifact=artifact,
                            path=gz_path,
                            size_bytes=stat.st_size,
                            mtime=stat.st_mtime,
                        )
                    )
                except OSError:
                    pass
        except OSError:
            pass
        entries.sort(key=lambda e: e.mtime, reverse=True)
        return entries

    def stats(self) -> dict[str, object]:
        """Return a summary dict suitable for human display."""
        entries = self.list_entries()
        total_bytes = sum(e.size_bytes for e in entries)
        artifact_counts: dict[str, int] = {}
        for entry in entries:
            artifact_counts[entry.artifact] = artifact_counts.get(entry.artifact, 0) + 1
        return {
            "entry_count": len(entries),
            "total_bytes": total_bytes,
            "total_mb": round(total_bytes / 1024**2, 2),
            "total_gb": round(total_bytes / 1024**3, 4),
            "cache_dir": str(self.cache_dir),
            "max_gb": self.max_bytes / 1024**3 if self.max_bytes > 0 else "unlimited",
            "ttl_days": self.ttl_seconds / 86400 if self.ttl_seconds > 0 else "none",
            "artifact_counts": artifact_counts,
        }

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def purge(self, older_than_days: float | None = None) -> int:
        """Delete cache entries.

        Parameters
        ----------
        older_than_days:
            If given, only delete entries older than this many days.
            If ``None``, delete all entries.

        Returns
        -------
        int
            Bytes freed.
        """
        freed = 0
        cutoff: float | None = None
        if older_than_days is not None:
            cutoff = time.time() - older_than_days * 86400
        for entry in self.list_entries():
            if cutoff is not None and entry.mtime > cutoff:
                continue
            try:
                freed += entry.size_bytes
                entry.path.unlink(missing_ok=True)
            except OSError:
                pass
        self._clean_empty_dirs()
        return freed

    def _clean_empty_dirs(self) -> None:
        """Remove empty shard directories."""
        try:
            for child in sorted(self.cache_dir.iterdir(), reverse=True):
                if child.is_dir():
                    try:
                        child.rmdir()  # no-op if not empty
                    except OSError:
                        pass
        except OSError:
            pass

    def _enforce_budget(self) -> int:
        """Remove oldest entries until total size is within ``max_bytes``."""
        if self.max_bytes <= 0:
            return 0
        entries = sorted(self.list_entries(), key=lambda e: e.mtime)  # oldest first
        total = sum(e.size_bytes for e in entries)
        freed = 0
        while total > self.max_bytes and entries:
            oldest = entries.pop(0)
            try:
                freed += oldest.size_bytes
                total -= oldest.size_bytes
                oldest.path.unlink(missing_ok=True)
            except OSError:
                pass
        return freed


# ------------------------------------------------------------------
# Artifact-specific key builders
# ------------------------------------------------------------------


def scene_events_key(input_path: Path, scene_threshold: float) -> str:
    return make_file_cache_key(
        input_path,
        "scene_events",
        settings={"scene_threshold": round(scene_threshold, 6)},
    )


def cropdetect_key(input_path: Path) -> str:
    return make_file_cache_key(input_path, "cropdetect")


def signalstats_key(input_path: Path, sample_fps: float) -> str:
    return make_file_cache_key(
        input_path,
        "signalstats",
        settings={"sample_fps": round(sample_fps, 6)},
    )


def whisper_key(
    input_path: Path,
    *,
    model_path: Path,
    language: str,
    audio_stream: int | None,
    vad: bool,
    vad_threshold: float,
) -> str:
    try:
        model_stat = model_path.stat()
        model_id = f"{model_path.resolve()}|{model_stat.st_size}|{model_stat.st_mtime}"
    except OSError:
        model_id = str(model_path.resolve())
    return make_file_cache_key(
        input_path,
        "whisper",
        settings={
            "model": model_id,
            "language": language,
            "audio_stream": audio_stream,
            "vad": vad,
            "vad_threshold": round(vad_threshold, 4),
        },
    )


def ocr_cues_key(input_path: Path, sample_fps: float, cue_threshold: float) -> str:
    return make_file_cache_key(
        input_path,
        "ocr_cues",
        settings={
            "sample_fps": round(sample_fps, 6),
            "cue_threshold": round(cue_threshold, 6),
        },
    )


def result_time_key(input_path: Path, sample_fps: float, tail_seconds: float) -> str:
    return make_file_cache_key(
        input_path,
        "result_time",
        settings={
            "sample_fps": round(sample_fps, 6),
            "tail_seconds": round(tail_seconds, 3),
        },
    )
