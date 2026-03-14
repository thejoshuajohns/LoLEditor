"""Prepare branded overlay assets for rendered videos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config.settings import ChampionContext, ContentConfig, OverlayConfig, UploadConfig
from ..rendering.overlays import (
    create_end_card_image,
    create_intro_card_image,
    create_kda_overlay_image,
    create_subscribe_animation_frames,
)


def build_overlay_bundle(
    *,
    champion: ChampionContext,
    content: ContentConfig,
    overlay_config: OverlayConfig,
    upload: UploadConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate reusable overlay assets and return a manifest."""

    assets_dir = output_dir / "artifacts" / "overlays"
    assets_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "enabled": True,
        "assets_dir": str(assets_dir),
        "items": [],
    }

    title_text = content.thumbnail_headline.strip() or content.title.strip()
    channel_name = upload.channel_name.strip() or champion.player_name.strip() or "LoLEditor"

    if overlay_config.enable_intro and champion.champion_name.strip():
        intro_path = assets_dir / "intro-card.png"
        create_intro_card_image(
            output_path=intro_path,
            champion_name=champion.champion_name,
            player_name=champion.player_name,
            rank=champion.rank,
            lane=champion.lane,
        )
        manifest["intro_card"] = str(intro_path)
        manifest["items"].append({"type": "intro_card", "path": str(intro_path)})

    if (
        overlay_config.enable_kda_overlay
        and champion.kills is not None
        and champion.deaths is not None
        and champion.assists is not None
    ):
        kda_path = assets_dir / "kda-overlay.png"
        create_kda_overlay_image(
            output_path=kda_path,
            kills=champion.kills,
            deaths=champion.deaths,
            assists=champion.assists,
            champion_name=champion.champion_name,
        )
        manifest["kda_overlay"] = str(kda_path)
        manifest["items"].append({"type": "kda_overlay", "path": str(kda_path)})

    if overlay_config.enable_outro:
        end_card_path = assets_dir / "end-card.png"
        create_end_card_image(
            output_path=end_card_path,
            channel_name=channel_name,
            subscribe_text=title_text or "SUBSCRIBE FOR MORE HIGHLIGHTS",
        )
        manifest["end_card"] = str(end_card_path)
        manifest["items"].append({"type": "end_card", "path": str(end_card_path)})

    if overlay_config.enable_subscribe_animation:
        subscribe_dir = assets_dir / "subscribe"
        frames = create_subscribe_animation_frames(output_dir=subscribe_dir)
        manifest["subscribe_frames"] = [str(path) for path in frames]
        manifest["items"].append(
            {
                "type": "subscribe_frames",
                "count": len(frames),
                "path": str(subscribe_dir),
            }
        )

    if overlay_config.enable_champion_portrait and champion.champion_png.strip():
        manifest["champion_portrait"] = champion.champion_png
        manifest["items"].append(
            {
                "type": "champion_portrait",
                "path": champion.champion_png,
            }
        )

    return manifest
