"""Generate publish-ready metadata from highlight analysis results."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..config.settings import ChampionContext, ContentConfig, OutputConfig, UploadConfig


def build_content_package(
    *,
    champion: ChampionContext,
    content: ContentConfig,
    output: OutputConfig,
    upload: UploadConfig,
    highlights: list[dict[str, Any]],
    edit_plan: dict[str, Any] | None,
    match_duration_seconds: float,
) -> dict[str, Any]:
    """Build title, description, tags, chapters, and summary copy."""

    summary = _summarize_highlights(highlights)
    chapters = _build_chapters(edit_plan, highlights, enabled=content.auto_generate_chapters)
    title_candidates = _build_title_candidates(
        champion=champion,
        summary=summary,
        highlight_count=len(highlights),
    )
    primary_title = (
        content.title.strip()
        or title_candidates[0]
        or "League of Legends Highlight Montage"
    )
    tags = (
        _dedupe_tags(content.tags)
        if content.tags and not content.auto_generate_tags
        else _build_tags(
            champion=champion,
            upload=upload,
            output=output,
            summary=summary,
        )
    )
    description = (
        content.description.strip()
        if content.description.strip() and not content.auto_generate_description
        else _build_description(
            champion=champion,
            title=primary_title,
            summary=summary,
            chapters=chapters,
            tags=tags,
            match_duration_seconds=match_duration_seconds,
        )
    )
    thumbnail_headline = content.thumbnail_headline.strip() or _build_thumbnail_headline(
        champion=champion,
        summary=summary,
    )

    return {
        "title": primary_title,
        "title_candidates": title_candidates,
        "description": description,
        "tags": tags,
        "chapters": chapters,
        "summary": summary,
        "thumbnail_headline": thumbnail_headline,
    }


def _summarize_highlights(highlights: list[dict[str, Any]]) -> dict[str, Any]:
    if not highlights:
        return {
            "top_event": "highlight montage",
            "event_counts": {},
            "top_labels": [],
            "average_score": 0.0,
        }

    event_counts = Counter(str(item.get("event_type", "highlight")) for item in highlights)
    top_labels = [
        str(item.get("label", "Highlight"))
        for item in sorted(highlights, key=lambda item: float(item.get("score", 0.0)), reverse=True)[:5]
    ]
    average_score = sum(float(item.get("score", 0.0)) for item in highlights) / len(highlights)
    top_event = event_counts.most_common(1)[0][0]
    return {
        "top_event": top_event,
        "event_counts": dict(event_counts),
        "top_labels": top_labels,
        "average_score": round(average_score, 4),
    }


def _build_title_candidates(
    *,
    champion: ChampionContext,
    summary: dict[str, Any],
    highlight_count: int,
) -> list[str]:
    champion_name = champion.champion_name.strip() or "League"
    rank = champion.rank.strip().title()
    lane = champion.lane.strip().title()
    patch = champion.patch.strip()
    top_event = str(summary.get("top_event", "highlight")).replace("_", " ").title()

    details = [part for part in (rank, lane) if part]
    detail_suffix = f" {' '.join(details)}" if details else ""
    patch_suffix = f" on Patch {patch}" if patch else ""

    candidates = [
        f"{champion_name}{detail_suffix} Highlights | {top_event} Carry Game{patch_suffix}",
        f"{champion_name} Montage | {highlight_count} Ranked Highlights{patch_suffix}",
        f"{champion_name} Outplays and Teamfights | Creator Cut{patch_suffix}",
    ]
    return [candidate.strip() for candidate in candidates]


def _build_tags(
    *,
    champion: ChampionContext,
    upload: UploadConfig,
    output: OutputConfig,
    summary: dict[str, Any],
) -> list[str]:
    base_tags = list(upload.default_tags)
    extras = [
        champion.champion_name,
        champion.player_name,
        champion.rank,
        champion.lane,
        champion.role,
        champion.patch and f"patch {champion.patch}",
        output.format.replace("_", " "),
    ]
    extras.extend(str(label) for label in summary.get("top_labels", [])[:3])
    extras.extend(str(event).replace("_", " ") for event in summary.get("event_counts", {}).keys())
    return _dedupe_tags(base_tags + extras)


def _build_description(
    *,
    champion: ChampionContext,
    title: str,
    summary: dict[str, Any],
    chapters: list[dict[str, Any]],
    tags: list[str],
    match_duration_seconds: float,
) -> str:
    meta_bits = []
    if champion.player_name:
        meta_bits.append(f"Player: {champion.player_name}")
    if champion.champion_name:
        meta_bits.append(f"Champion: {champion.champion_name}")
    if champion.lane:
        meta_bits.append(f"Lane: {champion.lane.title()}")
    if champion.rank:
        meta_bits.append(f"Rank: {champion.rank.title()}")
    if champion.patch:
        meta_bits.append(f"Patch: {champion.patch}")

    event_counts = summary.get("event_counts", {})
    event_bits = [
        f"{count} {str(event).replace('_', ' ')}"
        for event, count in sorted(event_counts.items(), key=lambda item: (-item[1], item[0]))[:4]
    ]
    overview = "This AI-generated LoL montage blends scene changes, combat spikes, HUD cues, transcripts, and audio energy to surface the best moments automatically."
    if event_bits:
        overview += " Detected focus: " + ", ".join(event_bits) + "."

    lines = [title, "", overview]
    if match_duration_seconds > 0:
        lines.append(f"Source match length: {_format_mmss(match_duration_seconds)}")
    if meta_bits:
        lines.append(" | ".join(meta_bits))

    if chapters:
        lines.extend(["", "Chapters"])
        lines.extend(
            f"{chapter['timestamp']} {chapter['label']}"
            for chapter in chapters
        )

    hashtag_line = " ".join(_tag_to_hashtag(tag) for tag in tags[:6] if tag.strip())
    if hashtag_line:
        lines.extend(["", hashtag_line])
    return "\n".join(lines).strip() + "\n"


def _build_chapters(
    edit_plan: dict[str, Any] | None,
    highlights: list[dict[str, Any]],
    *,
    enabled: bool,
) -> list[dict[str, Any]]:
    if not enabled:
        return []

    chapter_source = list((edit_plan or {}).get("clips", [])) or highlights
    chapters: list[dict[str, Any]] = [{"time_seconds": 0, "timestamp": "00:00", "label": "Intro"}]
    seen_times = {0}
    for item in chapter_source[:8]:
        start = int(float(item.get("start", item.get("timestamp", 0.0)) or 0.0))
        label = str(item.get("label", "Highlight")).strip() or "Highlight"
        if start in seen_times:
            continue
        seen_times.add(start)
        chapters.append(
            {
                "time_seconds": start,
                "timestamp": _format_mmss(start),
                "label": label,
            }
        )
    return chapters


def _build_thumbnail_headline(
    *,
    champion: ChampionContext,
    summary: dict[str, Any],
) -> str:
    champion_name = champion.champion_name.strip().upper()
    if summary.get("top_labels"):
        headline = str(summary["top_labels"][0]).upper()
    else:
        headline = "RANKED HIGHLIGHTS"
    return f"{champion_name} {headline}".strip()


def _dedupe_tags(tags: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in tags:
        tag = " ".join(str(raw).strip().split())
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(tag)
    return normalized[:18]


def _tag_to_hashtag(tag: str) -> str:
    cleaned = "".join(ch for ch in tag.title() if ch.isalnum())
    return f"#{cleaned}" if cleaned else ""


def _format_mmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
