from pathlib import Path
from tempfile import TemporaryDirectory

from league_video_editor.application.content import build_content_package
from league_video_editor.application.project_store import ProjectStore
from league_video_editor.config.settings import (
    ChampionContext,
    ContentConfig,
    OutputConfig,
    UploadConfig,
)
from league_video_editor.server.app import _refresh_content_package
from league_video_editor.server.app import _apply_clip_order, _apply_clip_updates, _recompute_plan_stats
from fastapi import HTTPException
from league_video_editor.core.highlight_detector import detect_scoreboard_spikes
from league_video_editor.models import TranscriptionCue


def test_content_package_generates_title_tags_and_chapters() -> None:
    package = build_content_package(
        champion=ChampionContext(
            champion_name="Zed",
            lane="mid",
            rank="diamond",
            patch="14.10",
            player_name="Creator",
        ),
        content=ContentConfig(),
        output=OutputConfig(),
        upload=UploadConfig(default_tags=["leagueoflegends", "lolhighlights"]),
        highlights=[
            {
                "timestamp": 92.0,
                "label": "Triple Kill Eruption",
                "event_type": "multi_kill",
                "score": 0.92,
            },
            {
                "timestamp": 248.0,
                "label": "Baron Swing",
                "event_type": "objective",
                "score": 0.84,
            },
        ],
        edit_plan={
            "clips": [
                {"start": 0.0, "label": "Intro"},
                {"start": 92.0, "label": "Triple Kill Eruption"},
                {"start": 248.0, "label": "Baron Swing"},
            ]
        },
        match_duration_seconds=1800.0,
    )

    assert package["title"]
    assert "Zed" in package["title"]
    assert package["tags"]
    assert "leagueoflegends" in [tag.lower() for tag in package["tags"]]
    assert package["chapters"][0]["timestamp"] == "00:00"
    assert "thumbnail_headline" in package


def test_scoreboard_spike_detection_picks_up_score_swings() -> None:
    cues = [
        TranscriptionCue(start=10.0, end=12.0, score=0.2, text="Score 3-2", keywords=()),
        TranscriptionCue(start=45.0, end=47.0, score=0.3, text="Score 6-3", keywords=()),
        TranscriptionCue(start=80.0, end=82.0, score=0.4, text="Score 8-7", keywords=()),
    ]

    events = detect_scoreboard_spikes(cues)

    assert len(events) >= 2
    assert all(event.event_type == "scoreboard_spike" for event in events)


def test_project_store_persists_projects() -> None:
    with TemporaryDirectory() as temp_dir:
        store = ProjectStore(Path(temp_dir))
        project = store.create_project(
            name="Test Project",
            input_path="/tmp/match.mp4",
            pipeline_stage_ids=["import", "rendering"],
        )
        loaded = store.load_project(project["id"])

        assert loaded is not None
        assert loaded["name"] == "Test Project"
        assert loaded["pipeline_state"]["import"]["status"] == "pending"


def test_project_store_deletes_nested_workspace_contents() -> None:
    with TemporaryDirectory() as temp_dir:
        store = ProjectStore(Path(temp_dir))
        project = store.create_project(
            name="Nested Project",
            input_path="/tmp/match.mp4",
            pipeline_stage_ids=["import"],
        )
        nested_file = Path(store.project_file(project["id"]).parent) / "artifacts" / "overlays" / "intro-card.png"
        nested_file.parent.mkdir(parents=True, exist_ok=True)
        nested_file.write_text("artifact", encoding="utf-8")

        deleted = store.delete_project(project["id"])

        assert deleted is True
        assert not Path(store.project_file(project["id"]).parent).exists()


def test_refresh_content_package_updates_chapters_after_timeline_change() -> None:
    project = {
        "config": {
            "champion": {
                "champion_name": "Zed",
            },
            "content": {
                "auto_generate_title": True,
                "auto_generate_description": True,
                "auto_generate_tags": True,
                "auto_generate_chapters": True,
                "title": "",
                "description": "",
                "tags": [],
                "thumbnail_headline": "",
            },
            "output": {
                "format": "youtube",
                "resolution": "1080p",
                "framerate": 60,
                "codec": "auto",
                "crf": 20,
                "preset": "fast",
                "profile": "balanced",
            },
            "upload": {
                "auto_upload": False,
                "privacy": "private",
                "category_id": "20",
                "channel_name": "",
                "default_tags": ["leagueoflegends"],
            },
        },
        "duration_seconds": 1200.0,
        "highlights": [
            {"timestamp": 120.0, "label": "Fight One", "event_type": "teamfight", "score": 0.8},
            {"timestamp": 360.0, "label": "Fight Two", "event_type": "teamfight", "score": 0.82},
        ],
        "edit_plan": {
            "clips": [
                {"start": 120.0, "end": 150.0, "duration": 30.0, "label": "Fight One"},
                {"start": 360.0, "end": 392.0, "duration": 32.0, "label": "Fight Two"},
            ]
        },
    }

    _refresh_content_package(project)

    chapters = project["content_package"]["chapters"]
    assert chapters[1]["timestamp"] == "02:00"
    assert chapters[2]["timestamp"] == "06:00"


def test_clip_updates_clamp_negative_duration_and_recompute_stats() -> None:
    plan = {
        "clips": [
            {"start": 10.0, "end": 20.0, "duration": 10.0, "label": "Fight One"},
            {"start": 30.0, "end": 42.0, "duration": 12.0, "label": "Fight Two"},
        ]
    }

    updated = _apply_clip_updates(plan, [type("ClipUpdateLike", (), {"index": 0, "start": 25.0, "end": None, "delete": False})()])
    updated = _recompute_plan_stats(updated, source_duration=120.0)

    assert updated["clips"][0]["start"] == 25.0
    assert updated["clips"][0]["end"] == 25.0
    assert updated["clips"][0]["duration"] == 0.0
    assert updated["coverage_ratio"] >= 0.0


def test_clip_reorder_rejects_duplicate_indexes() -> None:
    plan = {
        "clips": [
            {"start": 10.0, "end": 20.0, "duration": 10.0, "label": "Fight One"},
            {"start": 30.0, "end": 42.0, "duration": 12.0, "label": "Fight Two"},
        ]
    }

    try:
        _apply_clip_order(plan, [0, 0])
    except HTTPException as err:
        assert err.status_code == 400
    else:
        raise AssertionError("Expected duplicate clip order to be rejected")
