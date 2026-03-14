"""Persistent project workspaces for the desktop application."""

from __future__ import annotations

import json
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..config.settings import ProjectConfig


class ProjectStore:
    """Persist desktop projects to disk so jobs survive app restarts."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_project(
        self,
        *,
        name: str,
        input_path: str,
        output_dir: str = "",
        pipeline_stage_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        project_id = uuid.uuid4().hex[:8]
        workspace_dir = self.root_dir / project_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        final_output_dir = Path(output_dir) if output_dir else workspace_dir / "output"
        final_output_dir.mkdir(parents=True, exist_ok=True)

        config = ProjectConfig(
            input_path=input_path,
            output_dir=str(final_output_dir),
        )
        now = time.time()
        project = {
            "id": project_id,
            "name": name,
            "input_path": input_path,
            "workspace_dir": str(workspace_dir),
            "output_dir": str(final_output_dir),
            "status": "created",
            "created_at": now,
            "updated_at": now,
            "config": config.to_dict(),
            "highlights": [],
            "highlight_summary": {},
            "edit_plan": None,
            "content_package": None,
            "overlay_assets": {},
            "artifacts": {},
            "analysis": {},
            "logs": [],
            "vision_windows": [],
            "scene_events": [],
            "duration_seconds": 0.0,
            "pipeline_state": {
                stage_id: {"progress": 0.0, "status": "pending", "message": ""}
                for stage_id in (pipeline_stage_ids or [])
            },
        }
        self.save_project(project)
        return project

    def list_projects(self) -> list[dict[str, Any]]:
        projects: list[dict[str, Any]] = []
        for project_dir in sorted(self.root_dir.iterdir()):
            if not project_dir.is_dir():
                continue
            project = self.load_project(project_dir.name)
            if project is not None:
                projects.append(project)
        projects.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
        return projects

    def load_project(self, project_id: str) -> dict[str, Any] | None:
        path = self.project_file(project_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def save_project(self, project: dict[str, Any]) -> None:
        path = self.project_file(str(project["id"]))
        payload = deepcopy(project)
        payload["updated_at"] = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def delete_project(self, project_id: str) -> bool:
        project_dir = self.root_dir / project_id
        if not project_dir.exists():
            return False
        for child in sorted(
            project_dir.rglob("*"),
            key=lambda path: (len(path.parts), str(path)),
            reverse=True,
        ):
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                child.rmdir()
        project_dir.rmdir()
        return True

    def patch_project(self, project_id: str, patch: dict[str, Any]) -> dict[str, Any] | None:
        project = self.load_project(project_id)
        if project is None:
            return None
        _deep_merge(project, patch)
        self.save_project(project)
        return project

    def project_file(self, project_id: str) -> Path:
        return self.root_dir / project_id / "project.json"


def _deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
