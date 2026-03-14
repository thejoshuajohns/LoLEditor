"""YouTube description and title-idea generator."""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from .analyze import (
    _collect_local_ai_cues,
    _collect_ocr_cues,
    _compute_vision_window_scores,
    score_transcript_text,
)
from .ffmpeg_utils import (
    detect_scene_events_adaptive,
    extract_signalstats_samples,
    probe_duration_seconds,
)
from .models import (
    DEFAULT_AI_CUE_THRESHOLD,
    DEFAULT_OCR_CUE_SCORING,
    DEFAULT_OCR_CUE_THRESHOLD,
    DEFAULT_OCR_SAMPLE_FPS,
    DEFAULT_TESSERACT_BIN,
    DEFAULT_THUMBNAIL_SCENE_THRESHOLD,
    DEFAULT_THUMBNAIL_VISION_SAMPLE_FPS,
    DEFAULT_THUMBNAIL_VISION_STEP_SECONDS,
    DEFAULT_THUMBNAIL_VISION_WINDOW_SECONDS,
    DEFAULT_VISION_SAMPLE_FPS,
    DEFAULT_VISION_STEP_SECONDS,
    DEFAULT_VISION_WINDOW_SECONDS,
    DEFAULT_WHISPER_CPP_BIN,
    DEFAULT_WHISPER_LANGUAGE,
    DEFAULT_WHISPER_THREADS,
    DEFAULT_WHISPER_VAD,
    DEFAULT_WHISPER_VAD_THRESHOLD,
    EditorError,
    TranscriptionCue,
    VisionWindow,
)
from .watchability import _build_watchability_report


def _format_youtube_timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _keyword_to_hashtag(keyword: str) -> str:
    pieces = [p for p in re.split(r"[^a-zA-Z0-9]+", keyword.strip()) if p]
    if not pieces:
        return ""
    return "#" + "".join(p[:1].upper() + p[1:] for p in pieces)


def _cue_label(cue: TranscriptionCue, *, index: int = 0) -> str:
    """Generate a creative, YouTube-style chapter label for a cue.

    Avoids generic labels — aims for the punchy style seen on gaming channels.
    Uses index to vary phrasing when multiple cues share the same keyword.
    """
    kw = set(cue.keywords)
    if "pentakill" in kw:
        return ["Pentakill Rampage", "The PENTAKILL", "Five Down, None Standing"][index % 3]
    if "quadra kill" in kw:
        return ["Quadra Kill Eruption", "One Away From Penta", "Four Piece Combo"][index % 3]
    if "triple kill" in kw:
        return ["Triple Kill Showcase", "Three Piece Special", "Triple Threat"][index % 3]
    if "shutdown" in kw:
        return ["Shutdown Collected", "Bounty Cashed", "Big Shutdown Swing"][index % 3]
    if "ace" in kw:
        return ["Clean Ace to End It", "Full Team Wiped", "Ace and Push"][index % 3]
    if {"baron", "baron nashor", "nashor"} & kw:
        return ["Baron Fight Erupts", "Contest at the Pit", "Baron Stolen or Secured"][index % 3]
    if {"elder dragon"} & kw:
        return ["Elder Dragon Showdown", "Elder Decides Everything", "Soul Point Fight"][index % 3]
    if "dragon" in kw:
        return ["Dragon Contest", "Fight at the Drake", "Objective Secured"][index % 3]
    if {"nexus"} & kw:
        return ["The Final Collapse", "Nexus Goes Down", "Game Over"][index % 3]
    if {"victory", "defeat"} & kw:
        return ["End Screen Reached", "The Final Push", "Victory Lap"][index % 3]
    if {"turret", "tower"} & kw:
        return ["Turning Kills into Towers", "Structure Crumbles", "Tower Dive Pays Off"][index % 3]
    if {"inhibitor", "structure destroyed"} & kw:
        return ["Inhib Down, Map Opens", "Breaking the Base", "Objective Pressure"][index % 3]
    if {"teamfight", "fight"} & kw:
        return ["Outplay That Flipped the Game", "Chaotic Teamfight", "5v5 Erupts"][index % 3]
    if {"first blood"} & kw:
        return ["First Blood in the Shadows", "Opening Kill Secured", "Early Aggression"][index % 3]
    if {"killing spree", "rampage", "unstoppable", "godlike", "legendary"} & kw:
        return ["Spree Keeps Growing", "Unstoppable Force", "On a Tear"][index % 3]
    if {"outplay"} & kw:
        return ["Surgical Execution", "Mechanical Outplay", "Outplay That Flipped the Game"][index % 3]
    if {"gank", "dive"} & kw:
        return ["Punished on Cooldown", "Caught Slipping", "Gank Pays Off"][index % 3]
    if {"kill", "slain", "enemy slain"} & kw:
        return ["Surgical Execution", "Punished on Cooldown", "Caught Slipping"][index % 3]
    generic = [
        "High-Intensity Sequence",
        "Momentum Shift",
        "Pressure Applied",
        "Clean Rotation",
        "Map Control Play",
    ]
    return generic[index % len(generic)]


def _sanitize_blurb(text: str, *, max_chars: int = 88) -> str:
    compact = re.sub(r"\s+", " ", text).strip().replace("[", "").replace("]", "")
    if len(compact) <= max_chars:
        return compact
    return f"{compact[:max_chars - 1].rstrip()}…"


def _clean_moment_blurb(text: str, *, max_chars: int = 88) -> str | None:
    compact = _sanitize_blurb(text, max_chars=max_chars)
    if len(compact) < 16:
        return None
    alpha = sum(1 for c in compact if c.isalpha())
    alnum = sum(1 for c in compact if c.isalnum())
    if alnum <= 0 or alpha / max(1, len(compact)) < 0.48:
        return None
    if len(re.findall(r"[A-Za-z]{3,}", compact)) < 3:
        return None
    return compact


def _dedupe_ranked_cues(
    cues: list[TranscriptionCue],
    *,
    min_gap_seconds: float = 10.0,
) -> list[TranscriptionCue]:
    ranked = sorted(cues, key=lambda c: (-c.score, c.start))
    selected: list[TranscriptionCue] = []
    for cue in ranked:
        if any(abs(cue.center - e.center) < min_gap_seconds for e in selected):
            continue
        selected.append(cue)
    return sorted(selected, key=lambda c: c.start)


def _fallback_moment_points_from_vision(
    *,
    vision_windows: list[VisionWindow],
    max_moments: int,
) -> list[tuple[float, str, str]]:
    if max_moments <= 0:
        return []
    ranked = sorted(vision_windows, key=lambda w: (-w.score, w.start))
    selected: list[tuple[float, str, str]] = []
    for w in ranked:
        center = (w.start + w.end) / 2.0
        if any(abs(center - existing[0]) < 14.0 for existing in selected):
            continue
        selected.append(
            (center, "High-intensity sequence", f"Activity score {w.score:.2f} (motion={w.motion:.2f}).")
        )
        if len(selected) >= max_moments:
            break
    return sorted(selected, key=lambda item: item[0])


def _extract_moment_points(
    *,
    cues: list[TranscriptionCue],
    vision_windows: list[VisionWindow],
    max_moments: int,
) -> list[tuple[float, str, str]]:
    if max_moments <= 0:
        return []
    if not cues:
        return _fallback_moment_points_from_vision(
            vision_windows=vision_windows, max_moments=max_moments
        )
    selected: list[tuple[float, str, str]] = []
    label_counter = 0
    for cue in sorted(cues, key=lambda c: (-c.score, c.start)):
        if any(abs(cue.center - existing[0]) < 12.0 for existing in selected):
            continue
        label = _cue_label(cue, index=label_counter)
        blurb = _clean_moment_blurb(cue.text) or label
        selected.append((cue.center, label, blurb))
        label_counter += 1
        if len(selected) >= max_moments:
            break
    return sorted(selected, key=lambda item: item[0])


def _keyword_weights_from_cues(cues: list[TranscriptionCue]) -> dict[str, float]:
    weighted: Counter[str] = Counter()
    for cue in cues:
        for kw in set(cue.keywords):
            weighted[kw] += cue.score
    return dict(weighted)


def _topic_from_keywords(
    *,
    keyword_weights: dict[str, float],
    activity_profile: str,
) -> str:
    suppressed = {"slain", "kill", "enemy slain", "ally slain", "fight", "teamfight", "flash", "ultimate", "ult"}
    prioritized = {kw: w for kw, w in keyword_weights.items() if kw not in suppressed}
    top = (
        sorted(prioritized.items(), key=lambda item: (-item[1], item[0]))[0][0]
        if prioritized
        else (sorted(keyword_weights.items(), key=lambda item: (-item[1], item[0]))[0][0] if keyword_weights else "")
    )
    topic_map = {
        "pentakill": "PENTAKILL MOMENTS",
        "quadra kill": "QUADRA KILL CLUTCHES",
        "triple kill": "TRIPLE KILL SEQUENCES",
        "shutdown": "SHUTDOWN SWINGS",
        "ace": "ACE TEAMFIGHTS",
        "baron": "BARON FIGHTS",
        "baron nashor": "BARON NASHOR FIGHTS",
        "nashor": "BARON FIGHTS",
        "elder dragon": "ELDER DRAGON FIGHTS",
        "dragon": "DRAGON FIGHTS",
        "nexus": "GAME-ENDING PUSHES",
        "turret": "OBJECTIVE PUSHES",
        "tower": "OBJECTIVE PUSHES",
        "inhibitor": "OBJECTIVE PUSHES",
    }
    if top in topic_map:
        return topic_map[top]
    if top:
        return top.upper()
    if activity_profile == "high-action":
        return "INSANE TEAMFIGHTS"
    if activity_profile == "low-action":
        return "CLUTCH MACRO MOMENTS"
    return "RANKED HIGHLIGHTS"


def _build_ctr_title_candidates(
    *,
    champion: str,
    topic: str,
    youtube_score: float,
    moment_count: int,
    title_count: int,
) -> list[str]:
    base = champion.strip() or "LoL"
    intensity = "INSANE" if youtube_score >= 75 else "CLUTCH" if youtube_score >= 62 else "HIGHLIGHT"
    candidates = [
        f"{base.upper()} {intensity}: {topic} | League of Legends Highlights",
        f"{moment_count} CRAZY MOMENTS - {base.upper()} {topic}",
        f"{base.upper()} HARD CARRY? {topic} (LoL Ranked)",
        f"{base.upper()} MONTAGE: {topic} | Ranked Gameplay",
        f"{topic} with {base.upper()} - League Highlights",
    ]
    deduped: list[str] = []
    for c in candidates:
        text = re.sub(r"\s+", " ", c).strip()
        if text not in deduped:
            deduped.append(text)
        if len(deduped) >= max(1, title_count):
            break
    return deduped


def generate_youtube_description(
    *,
    input_path: Path,
    output_path: Path,
    champion: str,
    channel_name: str,
    scene_threshold: float = DEFAULT_THUMBNAIL_SCENE_THRESHOLD,
    vision_sample_fps: float = DEFAULT_VISION_SAMPLE_FPS,
    vision_window_seconds: float = DEFAULT_VISION_WINDOW_SECONDS,
    vision_step_seconds: float = DEFAULT_VISION_STEP_SECONDS,
    whisper_model: Path | None = None,
    whisper_bin: str = DEFAULT_WHISPER_CPP_BIN,
    whisper_language: str = DEFAULT_WHISPER_LANGUAGE,
    whisper_threads: int = DEFAULT_WHISPER_THREADS,
    whisper_audio_stream: int = -1,
    whisper_vad: bool = DEFAULT_WHISPER_VAD,
    whisper_vad_threshold: float = DEFAULT_WHISPER_VAD_THRESHOLD,
    whisper_vad_model: Path | None = None,
    ocr_cue_scoring: str = DEFAULT_OCR_CUE_SCORING,
    tesseract_bin: str = DEFAULT_TESSERACT_BIN,
    ocr_sample_fps: float = DEFAULT_OCR_SAMPLE_FPS,
    ocr_cue_threshold: float = DEFAULT_OCR_CUE_THRESHOLD,
    ai_cue_threshold: float = DEFAULT_AI_CUE_THRESHOLD,
    max_moments: int = 8,
    title_count: int = 3,
    _probe_fn=None,
    _detect_events_fn=None,
    _score_vision_fn=None,
    _collect_ai_fn=None,
    _collect_ocr_fn=None,
) -> dict[str, object]:
    _do_probe = _probe_fn if _probe_fn is not None else probe_duration_seconds
    _do_detect = _detect_events_fn if _detect_events_fn is not None else detect_scene_events_adaptive
    _do_ai = _collect_ai_fn if _collect_ai_fn is not None else _collect_local_ai_cues
    _do_ocr = _collect_ocr_fn if _collect_ocr_fn is not None else _collect_ocr_cues
    duration_seconds = _do_probe(input_path)
    events, scene_threshold_used = _do_detect(
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
    watchability_report = _build_watchability_report(
        duration_seconds=duration_seconds,
        events=events,
        vision_windows=vision_windows,
        scene_threshold_used=scene_threshold_used,
    )

    whisper_cues: list[TranscriptionCue] = []
    if whisper_model is not None:
        print("Running local AI transcription (whisper.cpp)...", file=sys.stderr, flush=True)
        try:
            whisper_cues = _do_ai(
                input_path=input_path,
                whisper_model=whisper_model,
                whisper_binary=whisper_bin,
                whisper_language=whisper_language,
                whisper_threads=whisper_threads,
                cue_threshold=ai_cue_threshold,
                whisper_audio_stream=whisper_audio_stream if whisper_audio_stream >= 0 else None,
                whisper_vad=whisper_vad,
                whisper_vad_threshold=whisper_vad_threshold,
                whisper_vad_model=whisper_vad_model,
            )
        except (EditorError, subprocess.CalledProcessError) as err:
            print(f"Warning: local AI transcription unavailable ({err}).", file=sys.stderr, flush=True)

    ocr_cues: list[TranscriptionCue] = []
    if ocr_cue_scoring == "auto":
        print("Running OCR cue detection (tesseract)...", file=sys.stderr, flush=True)
        try:
            ocr_cues = _do_ocr(
                input_path=input_path,
                tesseract_binary=tesseract_bin,
                sample_fps=ocr_sample_fps,
                cue_threshold=ocr_cue_threshold,
            )
        except (EditorError, subprocess.CalledProcessError) as err:
            print(f"Warning: OCR cue detection unavailable ({err}).", file=sys.stderr, flush=True)

    combined_cues = _dedupe_ranked_cues(whisper_cues + ocr_cues, min_gap_seconds=8.0)
    moment_points = _extract_moment_points(
        cues=combined_cues, vision_windows=vision_windows, max_moments=max_moments
    )
    keyword_weights = _keyword_weights_from_cues(combined_cues)
    activity_profile = str(watchability_report.get("activity_profile", "standard"))
    topic = _topic_from_keywords(keyword_weights=keyword_weights, activity_profile=activity_profile)
    youtube_score = float(watchability_report.get("youtube_score", 0.0))
    titles = _build_ctr_title_candidates(
        champion=champion,
        topic=topic,
        youtube_score=youtube_score,
        moment_count=len(moment_points),
        title_count=title_count,
    )

    # Build chapter labels — use champion name in the opener if available.
    if champion.strip():
        opener = f"{champion.strip()} Online"
    else:
        opener = "Game Starts"
    chapter_points: list[tuple[float, str]] = [(0.0, opener)]
    chapter_points.extend((m[0], m[1]) for m in moment_points if m[0] >= 3.0)
    if duration_seconds >= 120:
        chapter_points.append(
            (max(0.0, duration_seconds - min(45.0, duration_seconds * 0.08)), "The Final Push")
        )
    chapter_points.sort(key=lambda item: item[0])
    deduped_chapters: list[tuple[float, str]] = []
    for ct, cl in chapter_points:
        if deduped_chapters and abs(ct - deduped_chapters[-1][0]) < 8.0:
            continue
        deduped_chapters.append((ct, cl))
    chapters = [{"timestamp": _format_youtube_timestamp(ct), "label": cl} for ct, cl in deduped_chapters]

    hashtags = ["#leagueoflegends", "#lolhighlights", "#gaming"]
    if champion.strip():
        champ_tag = "#" + re.sub(r"[^a-zA-Z0-9]", "", champion.strip()).lower()
        if champ_tag and champ_tag not in hashtags:
            hashtags.append(champ_tag)
    hashtag_suppressed = {"slain", "kill", "enemy slain", "ally slain", "fight", "teamfight", "flash", "ultimate", "ult"}
    for kw, _ in sorted(keyword_weights.items(), key=lambda item: (-item[1], item[0])):
        if kw in hashtag_suppressed:
            continue
        tag = "#" + re.sub(r"[^a-zA-Z0-9]", "", kw.strip()).lower()
        if tag and len(tag) > 2 and tag not in hashtags:
            hashtags.append(tag)
        if len(hashtags) >= 9:
            break

    # Build description in the user's preferred clean style:
    # timestamps → CTA → hashtags
    description_lines: list[str] = []
    for chapter in chapters:
        description_lines.append(f"{chapter['timestamp']} \u2013 {chapter['label']}")
    description_lines.extend(
        [
            "",
            (
                f"Subscribe for more ranked highlights from {channel_name}"
                if channel_name.strip()
                else "Subscribe for more ranked highlights"
            ),
            "Like + comment your favorite moment so I can tune future edits",
            "",
            "  ".join(hashtags),
        ]
    )
    description_text = "\n".join(description_lines).strip()
    output_lines = ["Title Ideas:"]
    for i, title in enumerate(titles, start=1):
        output_lines.append(f"{i}. {title}")
    output_lines.extend(["", "Description:", description_text])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines).strip() + "\n", encoding="utf-8")

    return {
        "titles": titles,
        "description_text": description_text,
        "hashtags": hashtags,
        "chapters": chapters,
        "topic": topic,
        "youtube_score": watchability_report.get("youtube_score"),
        "watchability_score": watchability_report.get("watchability_score"),
        "highlight_quality_score": watchability_report.get("highlight_quality_score"),
        "activity_profile": watchability_report.get("activity_profile"),
        "scene_threshold_used": scene_threshold_used,
        "event_count": len(events),
        "moment_count": len(moment_points),
        "whisper_cue_count": len(whisper_cues),
        "ocr_cue_count": len(ocr_cues),
    }
