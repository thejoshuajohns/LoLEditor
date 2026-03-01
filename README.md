# League Video Editor

Create YouTube-ready videos from your League of Legends OBS recordings.

## What this does

- Detects high-action moments using scene-change analysis.
- Builds a highlight plan (`edit-plan.json`) that you can inspect/edit.
- Renders a highlight montage (`highlights.mp4`) with:
  - 1080p output
  - 60 FPS
  - H.264 video + AAC audio
  - audio loudness normalization for YouTube
- Can also export the full match in YouTube-ready format.

## Requirements

- Python 3.11+
- `ffmpeg` and `ffprobe` available on your `PATH`

On macOS with Homebrew:

```bash
brew install ffmpeg
```

## Install

From this project folder:

```bash
python3 -m pip install -e .
```

This gives you the command:

```bash
lol-video-editor
```

## Quick Start

Use `auto` first:

```bash
lol-video-editor auto /path/to/your-obs-recording.mp4 \
  --output highlights.mp4 \
  --plan edit-plan.json
```

Then upload `highlights.mp4` to YouTube.

## Commands

### 1) Analyze (build edit plan only)

```bash
lol-video-editor analyze /path/to/recording.mp4 \
  --plan edit-plan.json
```

Tune detection:

```bash
lol-video-editor analyze /path/to/recording.mp4 \
  --scene-threshold 0.30 \
  --clip-before 10 \
  --clip-after 14 \
  --min-gap-seconds 20 \
  --max-clips 24
```

Notes:
- Lower `--scene-threshold` = more moments detected.
- Higher `--min-gap-seconds` = fewer clips, less spam.

### 2) Render (from existing plan)

```bash
lol-video-editor render /path/to/recording.mp4 \
  --plan edit-plan.json \
  --output highlights.mp4
```

### 3) Auto (analyze + render)

```bash
lol-video-editor auto /path/to/recording.mp4 \
  --output highlights.mp4 \
  --plan edit-plan.json
```

### 4) Full match export (no auto clipping)

```bash
lol-video-editor full /path/to/recording.mp4 \
  --output full-match-youtube.mp4
```

## Editing the plan manually

After `analyze`, open `edit-plan.json` and adjust `segments`:

```json
{
  "segments": [
    { "start": 112.5, "end": 134.2 },
    { "start": 540.0, "end": 565.0 }
  ]
}
```

Then run `render` with that plan.

## Testing

Run:

```bash
python3 -m unittest discover -s tests
```

## Limitations

- Scene-change detection is generic; it is not game-event aware (kills/objectives).
- No GUI yet (CLI-only MVP).
- Requires local `ffmpeg`.

If you want, next step is a lightweight local web UI for timeline editing before render.

