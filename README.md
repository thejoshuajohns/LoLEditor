# League Video Editor

Create YouTube-ready videos from your League of Legends OBS recordings.

## What this does

- Detects high-action moments using scene-change + vision scoring.
- Optionally adds local AI transcript cues (Whisper.cpp) to prioritize real callouts:
  - kills, shutdowns, aces, objectives, nexus/end calls
- Builds a highlight plan (`edit-plan.json`) that you can inspect/edit.
- Anchors highlights to include:
  - detected in-game spawn/start (skips client/loading when detected)
  - end of the match (nexus finish coverage)
- Uses a balanced target runtime by default (about 20 minutes) with fewer, longer cuts.
- Prioritizes kill/assist/death context:
  - detects high-intensity combat windows and keeps lead-in/setup
  - detects gray-screen death transitions and keeps lead-in context
- Renders a highlight montage (`highlights.mp4`) with:
  - 1080p output
  - 60 FPS
  - H.264 video + AAC audio
  - audio loudness normalization for YouTube
- Can also export the full match in YouTube-ready format.

## Requirements

- Python 3.11+
- `ffmpeg` and `ffprobe` available on your `PATH`
- `Pillow` (installed automatically with this package)
- Optional for local AI mode: `whisper.cpp` (`whisper-cli`) + a local Whisper model file

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

Use local AI scoring (fully local/offline):

```bash
lol-video-editor auto /path/to/your-obs-recording.mp4 \
  --output highlights.mp4 \
  --plan edit-plan.json \
  --vision-scoring local-ai \
  --whisper-model /path/to/ggml-base.en.bin \
  --whisper-bin whisper-cli \
  --whisper-language en
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
  --max-clips 24 \
  --target-duration-ratio 0.66 \
  --target-duration-seconds 0 \
  --intro-seconds 120 \
  --outro-seconds 60
```

Notes:
- Lower `--scene-threshold` = more moments detected.
- Higher `--min-gap-seconds` = fewer clips, less spam.
- `--vision-scoring heuristic` (default) uses frame-motion + color heuristics to keep lane context while emphasizing combat/death moments.
- `--vision-scoring local-ai` combines the existing vision scoring with local Whisper.cpp transcript cue detection.
- `--whisper-model` is required in `local-ai` mode.
- `--whisper-audio-stream` lets you select OBS audio stream index for AI transcription (`-1` auto/default, `0` first audio stream).
- `--whisper-vad` / `--whisper-vad-threshold` tune speech segmentation for local transcription.
- `--ocr-cue-scoring auto` adds optional local OCR cue detection (requires `tesseract`) for HUD text like slain/objectives.
- `--ai-cue-threshold` controls how aggressive transcript cue selection is (lower = more AI cues).
- `--end-on-result` (default on) trims highlights to the detected `Victory/Defeat` end so post-game downtime is excluded.
- `--one-shot-smart` (default on) does single-pass adaptive tuning for better kill context + pacing without multi-candidate optimization.
- By default, highlights target ~2/3 of the source match duration (`--target-duration-ratio 0.6667`).
- Set `--target-duration-seconds` to a positive value to force an absolute target.
- Set both `--target-duration-seconds 0` and `--target-duration-ratio 0` for an uncapped target.

## Local AI setup (Whisper.cpp)

1. Install whisper.cpp (must provide `whisper-cli` on `PATH`).
2. Download a Whisper model (for example `base.en`).
3. Optional: install `tesseract` for OCR HUD cue extraction (`brew install tesseract` on macOS).
4. Run `analyze` or `auto` with:
   - `--vision-scoring local-ai`
   - `--whisper-model /path/to/model`
   - optional stream selection for OBS multi-track captures: `--whisper-audio-stream 0` (or `1`, `2`, etc.)
   - optional tuning: `--whisper-threads`, `--whisper-vad`, `--ai-cue-threshold`, `--ocr-cue-scoring auto`

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

Optimize automatically for the best blended YouTube score:

```bash
lol-video-editor auto /path/to/recording.mp4 \
  --output highlights.mp4 \
  --plan edit-plan.json \
  --auto-optimize \
  --optimize-candidates 6 \
  --optimize-metric youtube
```

`--auto-optimize` is slower (multiple candidate renders). If you want one-and-done speed, keep it off and rely on default `--one-shot-smart`.

### 4) Full match export (no auto clipping)

```bash
lol-video-editor full /path/to/recording.mp4 \
  --output full-match-youtube.mp4
```

### 5) Thumbnail generation

Auto-pick a high-action frame (recommended):

```bash
lol-video-editor thumbnail /path/to/video.mp4 \
  --output thumbnail.jpg
```

Use a manual frame timestamp:

```bash
lol-video-editor thumbnail /path/to/video.mp4 \
  --output thumbnail.jpg \
  --timestamp 512.4
```

Add champion art + headline text:

```bash
lol-video-editor thumbnail /path/to/video.mp4 \
  --output thumbnail.jpg \
  --champion-overlay /path/to/champion-transparent.png \
  --champion-scale 0.60 \
  --champion-anchor right \
  --headline "PENTA IN RANKED" \
  --headline-size 122 \
  --headline-color "#ffd400"
```

Useful options:
- `--width` / `--height` set output size (default `1280x720`)
- `--enhance` adds light color/sharpness boost (default on)
- `--auto-crop` removes common black borders before resize (default on)
- `--champion-overlay` overlays your transparent PNG champion art.
- `--headline` overlays text; use `\n` in the value for multi-line headlines.

### 6) YouTube description generator

Generate CTR-focused title ideas + a ready-to-paste YouTube description based on detected moments:

```bash
lol-video-editor description /path/to/highlights.mp4 \
  --output youtube-description.txt \
  --champion Zed \
  --channel-name "YourChannel"
```

Optional local AI cue enrichment:

```bash
lol-video-editor description /path/to/highlights.mp4 \
  --output youtube-description.txt \
  --champion Zed \
  --whisper-model /path/to/ggml-small.en.bin \
  --whisper-bin whisper-cli \
  --ocr-cue-scoring auto
```

### 7) Watchability analysis

```bash
lol-video-editor watchability /path/to/highlights.mp4 \
  --report watchability-report.json
```

Outputs:
- `watchability_score` (pace/activity oriented)
- `highlight_quality_score` (general highlight quality)
- blended `youtube_score` (profile-aware blend of both)

The report also includes an `activity_profile`, expected pace, blend weights, and improvement suggestions so low-action matches are graded more fairly.
The watchability pass auto-retries lower scene thresholds when no events are detected and ignores padded borders during frame sampling.

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

- Local AI mode depends on your local whisper.cpp/model setup and hardware speed.
- Cue quality depends on microphone/game audio clarity and language match.
- No GUI yet (CLI-only MVP).
- Requires local `ffmpeg`.

If you want, next step is a lightweight local web UI for timeline editing before render.
