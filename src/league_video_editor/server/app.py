"""FastAPI backend for the LoLEditor desktop application."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - import guard for optional UI stack
    raise ImportError(
        "The LoLEditor UI server requires FastAPI and uvicorn.\n"
        "Install them with: pip install 'league-video-editor[ui]'"
    ) from exc

from ..application.analysis import (
    collect_enrichment_signals,
    extract_vision_windows,
    resolve_signal_weights,
    serialize_analysis_bundle,
    serialize_cue,
    serialize_highlight,
    serialize_vision_window,
)
from ..application.content import build_content_package
from ..application.overlays import build_overlay_bundle
from ..application.production import produce_highlight_package
from ..application.project_store import ProjectStore
from ..config.settings import (
    ChampionContext,
    GlobalConfig,
    ProjectConfig,
    _dict_to_project_config,
    load_global_config,
)
from ..core.highlight_detector import run_highlight_detection
from ..core.smart_editor import generate_smart_edit_plan
from ..ffmpeg_utils import detect_scene_events_adaptive, probe_duration_seconds
from ..upload import upload_to_youtube

logger = logging.getLogger(__name__)

APP_ROOT = Path.cwd() / ".loleditor"
PROJECTS_ROOT = APP_ROOT / "projects"


app = FastAPI(
    title="LoLEditor API",
    description="AI-powered League of Legends highlight generator",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

project_store = ProjectStore(PROJECTS_ROOT)
_projects: dict[str, dict[str, Any]] = {}
_ws_connections: dict[str, WebSocket] = {}
_tasks: dict[str, asyncio.Task[Any]] = {}


class ProjectCreate(BaseModel):
    name: str
    input_path: str
    output_dir: str = ""


class ChampionInput(BaseModel):
    champion_name: str = ""
    champion_png: str = ""
    lane: str = ""
    role: str = ""
    player_name: str = ""
    rank: str = ""
    patch: str = ""
    kills: int | None = None
    deaths: int | None = None
    assists: int | None = None


class ContentInput(BaseModel):
    title: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    thumbnail_headline: str = ""
    auto_generate_title: bool = True
    auto_generate_description: bool = True
    auto_generate_tags: bool = True
    auto_generate_chapters: bool = True


class DetectionSettings(BaseModel):
    scene_threshold: float = 0.35
    killfeed_enabled: bool = True
    combat_intensity_enabled: bool = True
    objective_detection_enabled: bool = True
    scoreboard_detection_enabled: bool = True
    whisper_enabled: bool = True
    audio_excitement_enabled: bool = True
    max_highlights: int = 30


class RenderSettings(BaseModel):
    profile: str = "balanced"
    crf: int = 20
    preset: str = "fast"
    auto_crop: bool = True
    crossfade_seconds: float = 0.6
    enable_overlays: bool = True


class ClipUpdate(BaseModel):
    index: int
    start: float | None = None
    end: float | None = None
    delete: bool = False


class UploadSettings(BaseModel):
    title: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    privacy: str = "private"
    thumbnail_path: str = ""


PIPELINE_STAGES = [
    {"id": "import", "label": "Video Imported", "weight": 0.05},
    {"id": "scene_detection", "label": "Scene Detection", "weight": 0.15},
    {"id": "highlight_detection", "label": "Highlight Detection", "weight": 0.20},
    {"id": "transcript", "label": "Transcript Processing", "weight": 0.15},
    {"id": "clip_planning", "label": "Clip Planning", "weight": 0.10},
    {"id": "rendering", "label": "Rendering", "weight": 0.18},
    {"id": "encoding", "label": "Encoding", "weight": 0.07},
    {"id": "thumbnail", "label": "Thumbnail Creation", "weight": 0.05},
    {"id": "upload", "label": "Uploading", "weight": 0.05},
]


def _load_project(project_id: str) -> dict[str, Any]:
    project = _projects.get(project_id)
    if project is None:
        project = project_store.load_project(project_id)
        if project is not None:
            _projects[project_id] = project
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


def _save_project(project: dict[str, Any]) -> dict[str, Any]:
    _projects[str(project["id"])] = project
    project_store.save_project(project)
    return project


def _project_config(project: dict[str, Any]) -> ProjectConfig:
    return _dict_to_project_config(project.get("config", {}))


def _apply_clip_updates(plan: dict[str, Any], updates: list[ClipUpdate]) -> dict[str, Any]:
    """Apply timeline clip mutations safely."""

    clips = plan["clips"]
    for update in sorted(updates, key=lambda item: -item.index):
        if update.index < 0 or update.index >= len(clips):
            continue
        if update.delete:
            clips.pop(update.index)
            continue

        clip = clips[update.index]
        start = float(clip["start"])
        end = float(clip["end"])
        if update.start is not None:
            start = max(0.0, update.start)
        if update.end is not None:
            end = update.end
        end = max(start, end)

        clip["start"] = start
        clip["end"] = end
        clip["duration"] = end - start

    plan["clips"] = sorted(clips, key=lambda clip: clip["start"])
    return plan


def _apply_clip_order(plan: dict[str, Any], order: list[int]) -> dict[str, Any]:
    """Reorder clips and reject invalid or duplicate indexes."""

    clips = plan["clips"]
    expected = list(range(len(clips)))
    if sorted(order) != expected:
        raise HTTPException(status_code=400, detail="Order must contain each clip index exactly once")
    plan["clips"] = [clips[index] for index in order]
    return plan


def _recompute_plan_stats(plan: dict[str, Any], *, source_duration: float) -> dict[str, Any]:
    """Refresh aggregate edit-plan metrics after manual edits."""

    plan["total_duration"] = sum(float(clip["duration"]) for clip in plan["clips"])
    plan["highlight_count"] = len(plan["clips"])
    plan["avg_clip_duration"] = (
        plan["total_duration"] / len(plan["clips"]) if plan["clips"] else 0.0
    )
    plan["coverage_ratio"] = (
        plan["total_duration"] / source_duration if source_duration > 0 else 0.0
    )
    return plan


def _refresh_content_package(project: dict[str, Any]) -> None:
    """Regenerate derived publishing metadata after timeline edits."""

    config = _project_config(project)
    content_package = build_content_package(
        champion=config.champion,
        content=config.content,
        output=config.output,
        upload=config.upload,
        highlights=project.get("highlights", []),
        edit_plan=project.get("edit_plan"),
        match_duration_seconds=float(project.get("duration_seconds", 0.0) or 0.0),
    )
    config.content.thumbnail_headline = str(content_package.get("thumbnail_headline", "")).strip()
    project["config"] = config.to_dict()
    project["content_package"] = content_package
    project["highlight_summary"] = content_package.get("summary", {})


def _stage_progress(project: dict[str, Any], stage: str, progress: float, message: str = "") -> None:
    pipeline_state = project.setdefault("pipeline_state", {})
    pipeline_state[stage] = {
        "progress": round(progress, 4),
        "status": "completed" if progress >= 1.0 else ("running" if progress > 0 else "pending"),
        "message": message,
    }


async def send_progress(
    project_id: str,
    stage: str,
    progress: float,
    message: str = "",
    eta: float | None = None,
) -> None:
    try:
        project = _load_project(project_id)
    except HTTPException:
        project = None
    if project is not None:
        _stage_progress(project, stage, progress, message)
        _save_project(project)

    websocket = _ws_connections.get(project_id)
    if websocket is None:
        return
    try:
        await websocket.send_json(
            {
                "type": "progress",
                "project_id": project_id,
                "stage": stage,
                "progress": round(progress, 4),
                "message": message,
                "eta_seconds": eta,
                "timestamp": time.time(),
            }
        )
    except Exception:
        logger.warning("Failed to send progress update for %s", project_id)


async def send_log(project_id: str, level: str, message: str) -> None:
    try:
        project = _load_project(project_id)
    except HTTPException:
        project = None
    if project is not None:
        logs = project.setdefault("logs", [])
        logs.append(
            {
                "type": "log",
                "project_id": project_id,
                "level": level,
                "message": message,
                "timestamp": time.time(),
            }
        )
        project["logs"] = logs[-250:]
        _save_project(project)

    websocket = _ws_connections.get(project_id)
    if websocket is None:
        return
    try:
        await websocket.send_json(
            {
                "type": "log",
                "project_id": project_id,
                "level": level,
                "message": message,
                "timestamp": time.time(),
            }
        )
    except Exception:
        logger.warning("Failed to send log update for %s", project_id)


@app.get("/api/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "version": "2.1.0"}


@app.get("/api/config")
async def get_config() -> dict[str, Any]:
    return load_global_config().to_dict()


@app.post("/api/config")
async def save_config(config: dict[str, Any]) -> dict[str, str]:
    global_config = load_global_config()
    for key, value in config.items():
        if hasattr(global_config, key):
            setattr(global_config, key, value)
    global_config.save()
    return {"status": "saved"}


@app.post("/api/projects")
async def create_project(req: ProjectCreate) -> dict[str, Any]:
    input_path = Path(req.input_path).expanduser()
    if not input_path.exists():
        raise HTTPException(status_code=400, detail=f"Input file not found: {req.input_path}")

    project = project_store.create_project(
        name=req.name,
        input_path=str(input_path),
        output_dir=req.output_dir,
        pipeline_stage_ids=[stage["id"] for stage in PIPELINE_STAGES],
    )
    project.setdefault("logs", [])
    _save_project(project)
    return project


@app.get("/api/projects")
async def list_projects() -> list[dict[str, Any]]:
    projects = project_store.list_projects()
    for project in projects:
        _projects[str(project["id"])] = project
    return projects


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str) -> dict[str, Any]:
    return _load_project(project_id)


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str) -> dict[str, str]:
    if project_id in _tasks and not _tasks[project_id].done():
        _tasks[project_id].cancel()
    deleted = project_store.delete_project(project_id)
    _projects.pop(project_id, None)
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"status": "deleted"}


@app.put("/api/projects/{project_id}/champion")
async def update_champion(project_id: str, champion: ChampionInput) -> dict[str, str]:
    project = _load_project(project_id)
    project.setdefault("config", {})["champion"] = champion.model_dump()
    _save_project(project)
    return {"status": "updated"}


@app.put("/api/projects/{project_id}/content")
async def update_content(project_id: str, content: ContentInput) -> dict[str, str]:
    project = _load_project(project_id)
    project.setdefault("config", {})["content"] = content.model_dump()
    _save_project(project)
    return {"status": "updated"}


@app.post("/api/projects/{project_id}/analyze")
async def start_analysis(project_id: str, settings: DetectionSettings | None = None) -> dict[str, str]:
    project = _load_project(project_id)
    if project.get("status") == "analyzing":
        raise HTTPException(status_code=409, detail="Analysis already in progress")

    project["status"] = "analyzing"
    _save_project(project)
    task = asyncio.create_task(_run_analysis_pipeline(project_id, settings or DetectionSettings()))
    _tasks[project_id] = task
    return {"status": "started", "project_id": project_id}


async def _run_analysis_pipeline(project_id: str, settings: DetectionSettings) -> None:
    project = _load_project(project_id)
    input_path = Path(project["input_path"])
    config = _project_config(project)
    config.detection.scene_threshold = settings.scene_threshold
    config.detection.killfeed_enabled = settings.killfeed_enabled
    config.detection.combat_intensity_enabled = settings.combat_intensity_enabled
    config.detection.objective_detection_enabled = settings.objective_detection_enabled
    config.detection.scoreboard_detection_enabled = settings.scoreboard_detection_enabled
    config.detection.whisper_enabled = settings.whisper_enabled
    config.detection.audio_excitement_enabled = settings.audio_excitement_enabled
    config.editing.max_clips = settings.max_highlights
    config.output_dir = project["output_dir"]
    project["config"] = config.to_dict()
    _save_project(project)

    try:
        await send_progress(project_id, "import", 0.2, "Validating source recording...")
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        duration_seconds = await asyncio.to_thread(probe_duration_seconds, input_path)
        project["duration_seconds"] = duration_seconds
        await send_progress(project_id, "import", 1.0, f"Loaded {input_path.name} ({duration_seconds:.0f}s)")

        await send_progress(project_id, "scene_detection", 0.1, "Detecting scene transitions...")
        scene_events, threshold_used = await asyncio.to_thread(
            detect_scene_events_adaptive,
            input_path,
            config.detection.scene_threshold,
            duration_seconds=duration_seconds,
            show_progress=False,
        )
        project["scene_events"] = scene_events
        project.setdefault("analysis", {})["scene_threshold_used"] = threshold_used
        await send_progress(project_id, "scene_detection", 1.0, f"Found {len(scene_events)} scene events")

        await send_progress(project_id, "highlight_detection", 0.2, "Scoring visual intensity...")
        vision_windows = await asyncio.to_thread(
            extract_vision_windows,
            input_path=input_path,
            scene_events=scene_events,
            duration_seconds=duration_seconds,
            sample_fps=1.0,
        )
        project["vision_windows"] = [serialize_vision_window(item) for item in vision_windows]
        await send_progress(project_id, "highlight_detection", 0.45, "Visual gameplay map built")

        await send_progress(project_id, "transcript", 0.1, "Collecting transcript, OCR, and audio cues...")
        enrichment = await asyncio.to_thread(
            collect_enrichment_signals,
            input_path=input_path,
            duration_seconds=duration_seconds,
            detection=config.detection,
        )
        whisper_cues = enrichment.get("whisper_cues") or []
        ocr_cues = enrichment.get("ocr_cues") or []
        audio_rms_samples = enrichment.get("audio_rms_samples") or []
        project["analysis"]["whisper_cues"] = [serialize_cue(item) for item in whisper_cues]
        project["analysis"]["ocr_cues"] = [serialize_cue(item) for item in ocr_cues]
        await send_progress(
            project_id,
            "transcript",
            1.0,
            f"Signals ready: {len(whisper_cues)} transcript, {len(ocr_cues)} OCR, {len(audio_rms_samples)} audio windows",
        )

        await send_progress(project_id, "highlight_detection", 0.7, "Ranking highlights with weighted multi-signal scoring...")
        highlights = await asyncio.to_thread(
            run_highlight_detection,
            scene_events=scene_events,
            vision_windows=vision_windows,
            whisper_cues=whisper_cues,
            ocr_cues=ocr_cues,
            audio_rms_samples=audio_rms_samples,
            weights=resolve_signal_weights(config.detection),
            max_highlights=config.editing.max_clips,
            duration_seconds=duration_seconds,
        )
        project["highlights"] = [serialize_highlight(item) for item in highlights]
        await send_progress(project_id, "highlight_detection", 1.0, f"Ranked {len(highlights)} highlight candidates")

        await send_progress(project_id, "clip_planning", 0.25, "Generating smart edit plan...")
        edit_plan = await asyncio.to_thread(
            generate_smart_edit_plan,
            highlights,
            source_duration=duration_seconds,
            target_duration=config.editing.target_duration_seconds,
            min_clip_seconds=config.editing.min_clip_seconds,
            max_clip_seconds=config.editing.max_clip_seconds,
            pre_fight_context=config.editing.pre_fight_context_seconds,
            post_fight_aftermath=config.editing.post_fight_aftermath_seconds,
            dynamic_length=config.editing.dynamic_length_enabled,
            retain_death_context=config.editing.retain_death_context,
            min_gap_between_clips=config.editing.min_gap_between_clips,
            max_clips=config.editing.max_clips,
        )
        project["edit_plan"] = edit_plan.to_dict()

        await send_progress(project_id, "clip_planning", 0.65, "Generating title, description, tags, and chapters...")
        content_package = await asyncio.to_thread(
            build_content_package,
            champion=config.champion,
            content=config.content,
            output=config.output,
            upload=config.upload,
            highlights=project["highlights"],
            edit_plan=project["edit_plan"],
            match_duration_seconds=duration_seconds,
        )
        config.content.thumbnail_headline = str(content_package.get("thumbnail_headline", "")).strip()
        project["config"] = config.to_dict()
        project["content_package"] = content_package
        project["highlight_summary"] = content_package.get("summary", {})

        overlay_bundle = await asyncio.to_thread(
            build_overlay_bundle,
            champion=config.champion,
            content=config.content,
            overlay_config=config.overlays,
            upload=config.upload,
            output_dir=Path(project["output_dir"]),
        )
        project["overlay_assets"] = overlay_bundle
        project["analysis"]["bundle_preview"] = serialize_analysis_bundle(
            bundle=type(
                "BundlePreview",
                (),
                {
                    "duration_seconds": duration_seconds,
                    "scene_threshold_used": threshold_used,
                    "scene_events": scene_events,
                    "vision_windows": vision_windows,
                    "whisper_cues": whisper_cues,
                    "ocr_cues": ocr_cues,
                    "highlights": highlights,
                    "edit_plan": edit_plan,
                    "content_package": content_package,
                    "overlay_bundle": overlay_bundle,
                },
            )()
        )

        project["status"] = "analyzed"
        _save_project(project)
        await send_progress(
            project_id,
            "clip_planning",
            1.0,
            f"Edit plan ready: {edit_plan.highlight_count} clips / {edit_plan.total_duration:.0f}s",
        )
        await send_log(project_id, "info", "Analysis complete. Timeline, metadata, and overlays are ready.")

    except asyncio.CancelledError:
        project["status"] = "cancelled"
        _save_project(project)
        await send_log(project_id, "warning", "Analysis cancelled")
        raise
    except Exception as err:
        logger.exception("Analysis pipeline failed for %s", project_id)
        project["status"] = "error"
        _save_project(project)
        await send_log(project_id, "error", f"Analysis failed: {err}")


@app.post("/api/projects/{project_id}/render")
async def start_render(project_id: str, settings: RenderSettings | None = None) -> dict[str, str]:
    project = _load_project(project_id)
    if not project.get("edit_plan"):
        raise HTTPException(status_code=400, detail="No edit plan. Run analysis first.")

    project["status"] = "rendering"
    _save_project(project)
    task = asyncio.create_task(_run_render_pipeline(project_id, settings or RenderSettings()))
    _tasks[project_id] = task
    return {"status": "started"}


async def _run_render_pipeline(project_id: str, settings: RenderSettings) -> None:
    project = _load_project(project_id)
    config = _project_config(project)
    input_path = Path(project["input_path"])
    output_dir = Path(project["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        await send_progress(project_id, "rendering", 0.1, "Writing edit plan and preparing assets...")
        plan_path = output_dir / "edit-plan.json"
        plan_payload = {
            "input": str(input_path),
            "duration_seconds": project.get("duration_seconds", 0),
            "segments": [
                {
                    "start": clip["start"],
                    "end": clip["end"],
                    "duration": clip["duration"],
                    "label": clip.get("label", ""),
                }
                for clip in project["edit_plan"]["clips"]
            ],
            "settings": {
                "profile": settings.profile,
                "enable_overlays": settings.enable_overlays,
            },
        }
        plan_path.write_text(json.dumps(plan_payload, indent=2) + "\n", encoding="utf-8")

        await send_progress(project_id, "rendering", 0.45, "Rendering montage and applying branding...")
        artifacts = await asyncio.to_thread(
            produce_highlight_package,
            input_path=input_path,
            plan_path=plan_path,
            project_config=config,
            output_dir=output_dir,
            render_profile=settings.profile,
            crf=settings.crf,
            preset=settings.preset,
            auto_crop=settings.auto_crop,
            crossfade_seconds=settings.crossfade_seconds,
        )
        project["artifacts"] = artifacts
        project["overlay_assets"] = artifacts.get("overlay_manifest", project.get("overlay_assets", {}))
        await send_progress(project_id, "rendering", 1.0, "Montage render complete")
        await send_progress(project_id, "encoding", 1.0, "Packaging final deliverables")
        await send_progress(project_id, "thumbnail", 1.0, "Thumbnail created")

        project["output_path"] = artifacts.get("final_video", "")
        project["thumbnail_path"] = artifacts.get("thumbnail", "")
        project["status"] = "rendered"
        _save_project(project)
        await send_log(project_id, "info", "Render complete. Final video, thumbnail, and overlay assets are ready.")

    except asyncio.CancelledError:
        project["status"] = "cancelled"
        _save_project(project)
        await send_log(project_id, "warning", "Render cancelled")
        raise
    except Exception as err:
        logger.exception("Render failed for %s", project_id)
        project["status"] = "error"
        _save_project(project)
        await send_log(project_id, "error", f"Render failed: {err}")


@app.post("/api/projects/{project_id}/upload")
async def start_upload(project_id: str, settings: UploadSettings) -> dict[str, str]:
    project = _load_project(project_id)
    output_path = Path(project.get("output_path", ""))
    if not output_path.exists():
        raise HTTPException(status_code=400, detail="No rendered output found. Render the video first.")

    project["status"] = "uploading"
    _save_project(project)
    task = asyncio.create_task(_run_upload_pipeline(project_id, settings))
    _tasks[project_id] = task
    return {"status": "started"}


async def _run_upload_pipeline(project_id: str, settings: UploadSettings) -> None:
    project = _load_project(project_id)
    output_path = Path(project["output_path"])
    thumbnail_path = Path(settings.thumbnail_path) if settings.thumbnail_path else Path(project.get("thumbnail_path", ""))

    try:
        await send_progress(project_id, "upload", 0.15, "Starting YouTube upload flow...")
        response = await asyncio.to_thread(
            upload_to_youtube,
            output_path,
            title=settings.title,
            description=settings.description,
            tags=settings.tags,
            privacy=settings.privacy,
            thumbnail_path=thumbnail_path if thumbnail_path.exists() else None,
        )
        project["upload_result"] = response
        project["status"] = "uploaded"
        _save_project(project)
        await send_progress(project_id, "upload", 1.0, f"Uploaded successfully: {response.get('id', '')}")
        await send_log(project_id, "info", "YouTube upload complete")
    except Exception as err:
        logger.exception("Upload failed for %s", project_id)
        project["status"] = "error"
        _save_project(project)
        await send_log(project_id, "error", f"Upload failed: {err}")


@app.get("/api/projects/{project_id}/timeline")
async def get_timeline(project_id: str) -> dict[str, Any]:
    project = _load_project(project_id)
    return {
        "duration_seconds": project.get("duration_seconds", 0),
        "edit_plan": project.get("edit_plan"),
        "highlights": project.get("highlights", []),
        "vision_windows": project.get("vision_windows", []),
        "content_package": project.get("content_package"),
        "overlay_assets": project.get("overlay_assets", {}),
    }


@app.put("/api/projects/{project_id}/timeline/clips")
async def update_clips(project_id: str, updates: list[ClipUpdate]) -> dict[str, Any]:
    project = _load_project(project_id)
    plan = project.get("edit_plan")
    if not plan:
        raise HTTPException(status_code=400, detail="No edit plan")

    plan = _apply_clip_updates(plan, updates)
    plan = _recompute_plan_stats(plan, source_duration=float(project.get("duration_seconds", 0.0) or 0.0))
    project["edit_plan"] = plan
    _refresh_content_package(project)
    _save_project(project)
    return plan


@app.post("/api/projects/{project_id}/timeline/reorder")
async def reorder_clips(project_id: str, order: list[int]) -> dict[str, Any]:
    project = _load_project(project_id)
    plan = project.get("edit_plan")
    if not plan:
        raise HTTPException(status_code=400, detail="No edit plan")
    if len(order) != len(plan["clips"]):
        raise HTTPException(status_code=400, detail="Order length must match clip count")
    plan = _apply_clip_order(plan, order)
    plan = _recompute_plan_stats(plan, source_duration=float(project.get("duration_seconds", 0.0) or 0.0))
    project["edit_plan"] = plan
    _refresh_content_package(project)
    _save_project(project)
    return plan


@app.get("/api/pipeline-stages")
async def get_pipeline_stages() -> list[dict[str, Any]]:
    return PIPELINE_STAGES


@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: str) -> None:
    await websocket.accept()
    _ws_connections[project_id] = websocket
    try:
        project = _load_project(project_id)
        for log in project.get("logs", [])[-50:]:
            await websocket.send_json(log)
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            if payload.get("type") == "cancel":
                task = _tasks.get(project_id)
                if task and not task.done():
                    task.cancel()
                    await websocket.send_json({"type": "cancelled"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for project %s", project_id)
    finally:
        _ws_connections.pop(project_id, None)


def run_server(host: str = "127.0.0.1", port: int = 8420) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError("uvicorn is required: pip install uvicorn") from exc

    print(f"\n  LoLEditor Server v2.1.0")
    print(f"  API:  http://{host}:{port}/api/health")
    print(f"  Docs: http://{host}:{port}/docs\n")
    uvicorn.run(app, host=host, port=port, log_level="info")
