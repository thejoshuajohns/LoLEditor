import argparse
import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from league_video_editor.cli import (
    EditorError,
    Segment,
    TranscriptionCue,
    VisionWindow,
    _boost_vision_windows_with_ai_cues,
    _collect_local_ai_cues,
    _collect_ocr_cues,
    _clean_moment_blurb,
    analyze_watchability,
    _build_filter_complex,
    _build_watchability_report,
    _compute_vision_window_scores,
    _detect_combat_cues,
    _detect_death_cues,
    _detect_watchability_crop_filter,
    _estimate_render_duration,
    _extract_loudnorm_json,
    _extract_audio_for_whisper,
    _parse_whisper_json_cues,
    _rank_vision_candidates,
    _score_transcript_text,
    _looks_like_loudnorm_nonfinite_error,
    _probe_duration_seconds,
    _validate_cli_options,
    _video_codec_args,
    _video_postprocess_filter,
    build_segments,
    detect_scene_events,
    detect_scene_events_adaptive,
    build_arg_parser,
    generate_youtube_description,
    generate_thumbnail,
    merge_segments,
    read_plan,
    render_highlights,
    sample_evenly,
    transcode_full_match,
    write_plan,
)


class SegmentLogicTests(unittest.TestCase):
    def test_sample_evenly_reduces_list(self) -> None:
        values = [10, 20, 30, 40, 50, 60]
        sampled = sample_evenly(values, 3)
        self.assertEqual(sampled, [10, 30, 60])

    def test_merge_segments_combines_overlaps(self) -> None:
        merged = merge_segments(
            [
                Segment(0.0, 12.0),
                Segment(11.9, 20.0),
                Segment(25.0, 30.0),
            ]
        )
        self.assertEqual(merged, [Segment(0.0, 20.0), Segment(25.0, 30.0)])

    def test_estimate_render_duration_with_crossfade(self) -> None:
        duration = _estimate_render_duration(
            [Segment(0.0, 12.0), Segment(20.0, 35.0), Segment(60.0, 72.0)],
            crossfade_seconds=1.0,
        )
        self.assertAlmostEqual(duration, 37.0)

    def test_build_segments_uses_fallback_when_no_events(self) -> None:
        segments, used_fallback = build_segments(
            [],
            duration_seconds=1200.0,
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=18.0,
            max_clips=20,
        )
        self.assertTrue(used_fallback)
        self.assertGreaterEqual(len(segments), 2)
        self.assertAlmostEqual(segments[0].start, 0.0)
        self.assertAlmostEqual(segments[-1].end, 1200.0)

    def test_build_segments_targets_approximately_ten_minutes(self) -> None:
        events = [float(value) for value in range(60, 1140, 24)]
        segments, used_fallback = build_segments(
            events,
            duration_seconds=1200.0,
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=18.0,
            max_clips=30,
            target_duration_seconds=600.0,
            intro_seconds=45.0,
            outro_seconds=60.0,
        )
        self.assertFalse(used_fallback)
        self.assertAlmostEqual(segments[0].start, 0.0)
        self.assertAlmostEqual(segments[-1].end, 1116.0)
        total_duration = sum(segment.duration for segment in segments)
        self.assertGreaterEqual(total_duration, 560.0)
        self.assertLessEqual(total_duration, 640.0)

    def test_build_segments_expands_clip_windows_to_hit_target_with_limited_clips(self) -> None:
        events = [float(value) for value in range(60, 1140, 24)]
        segments, used_fallback = build_segments(
            events,
            duration_seconds=1200.0,
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=18.0,
            max_clips=20,
            target_duration_seconds=600.0,
            intro_seconds=45.0,
            outro_seconds=60.0,
        )
        self.assertFalse(used_fallback)
        total_duration = sum(segment.duration for segment in segments)
        self.assertGreaterEqual(total_duration, 560.0)

    def test_build_segments_uncapped_target_produces_longer_output(self) -> None:
        events = [float(value) for value in range(60, 1140, 24)]
        capped_segments, _ = build_segments(
            events,
            duration_seconds=1200.0,
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=18.0,
            max_clips=14,
            target_duration_seconds=600.0,
            intro_seconds=45.0,
            outro_seconds=60.0,
        )
        uncapped_segments, _ = build_segments(
            events,
            duration_seconds=1200.0,
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=18.0,
            max_clips=14,
            target_duration_seconds=0.0,
            intro_seconds=45.0,
            outro_seconds=60.0,
        )
        self.assertGreaterEqual(
            sum(segment.duration for segment in uncapped_segments),
            sum(segment.duration for segment in capped_segments),
        )

    def test_compute_vision_window_scores_penalizes_idle_and_low_saturation(self) -> None:
        frame_samples: list[tuple[float, float, float]] = []
        for second in range(0, 30):
            frame_samples.append((float(second), 10.0, 70.0))
        for second in range(30, 60):
            frame_samples.append((float(second), 0.5, 8.0))

        windows = _compute_vision_window_scores(
            frame_samples=frame_samples,
            events=[5.0, 10.0, 18.0, 22.0],
            duration_seconds=60.0,
            window_seconds=10.0,
            step_seconds=10.0,
        )
        self.assertEqual(len(windows), 6)
        self.assertGreater(windows[0].score, windows[-1].score)
        self.assertGreater(windows[1].score, windows[4].score)

    def test_build_segments_uses_vision_windows_when_available(self) -> None:
        windows = [
            VisionWindow(start=160.0, end=220.0, score=0.98, motion=12.0, saturation=60.0, scene_density=0.2),
            VisionWindow(start=230.0, end=290.0, score=0.94, motion=10.0, saturation=58.0, scene_density=0.18),
            VisionWindow(start=600.0, end=650.0, score=0.10, motion=1.0, saturation=10.0, scene_density=0.01),
        ]
        segments, _ = build_segments(
            [90.0, 300.0, 610.0, 900.0],
            duration_seconds=1200.0,
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=18.0,
            max_clips=4,
            target_duration_seconds=360.0,
            intro_seconds=45.0,
            outro_seconds=60.0,
            vision_windows=windows,
        )
        self.assertGreaterEqual(segments[0].start, 150.0)
        self.assertLessEqual(segments[0].start, 170.0)
        middle_segments = [segment for segment in segments if segment.start > segments[0].end and segment.end < 1140.0]
        self.assertGreaterEqual(len(middle_segments), 1)
        self.assertTrue(any(segment.start <= 250.0 <= segment.end for segment in middle_segments))

    def test_detect_death_cues_finds_gray_screen_transitions(self) -> None:
        windows = [
            VisionWindow(start=0.0, end=20.0, score=0.65, motion=8.0, saturation=45.0, scene_density=0.2),
            VisionWindow(start=20.0, end=40.0, score=0.62, motion=7.5, saturation=44.0, scene_density=0.2),
            VisionWindow(start=40.0, end=60.0, score=0.22, motion=2.4, saturation=9.0, scene_density=0.03),
            VisionWindow(start=60.0, end=80.0, score=0.20, motion=2.1, saturation=8.5, scene_density=0.02),
            VisionWindow(start=80.0, end=100.0, score=0.60, motion=7.2, saturation=42.0, scene_density=0.18),
            VisionWindow(start=100.0, end=120.0, score=0.21, motion=2.0, saturation=8.0, scene_density=0.02),
            VisionWindow(start=120.0, end=140.0, score=0.19, motion=1.8, saturation=7.5, scene_density=0.02),
        ]
        cues = _detect_death_cues(windows, min_spacing_seconds=20.0)
        self.assertEqual(len(cues), 2)
        self.assertLess(cues[0], 40.0)
        self.assertLess(cues[1], 100.0)

    def test_detect_combat_cues_finds_high_activity_spikes(self) -> None:
        windows = [
            VisionWindow(start=0.0, end=20.0, score=0.30, motion=3.0, saturation=30.0, scene_density=0.04),
            VisionWindow(start=20.0, end=40.0, score=0.85, motion=12.0, saturation=42.0, scene_density=0.26),
            VisionWindow(start=40.0, end=60.0, score=0.72, motion=9.0, saturation=40.0, scene_density=0.19),
            VisionWindow(start=60.0, end=80.0, score=0.28, motion=3.2, saturation=28.0, scene_density=0.03),
            VisionWindow(start=80.0, end=100.0, score=0.82, motion=11.0, saturation=43.0, scene_density=0.24),
            VisionWindow(start=100.0, end=120.0, score=0.75, motion=9.2, saturation=41.0, scene_density=0.18),
        ]
        cues = _detect_combat_cues(windows, min_spacing_seconds=20.0)
        self.assertGreaterEqual(len(cues), 2)
        self.assertLess(cues[0], 20.0)
        self.assertLess(cues[1], 80.0)

    def test_rank_vision_candidates_respects_requested_min_gap(self) -> None:
        windows = [
            VisionWindow(start=0.0, end=10.0, score=0.95, motion=8.0, saturation=30.0, scene_density=0.20),
            VisionWindow(start=10.0, end=20.0, score=0.90, motion=7.8, saturation=30.0, scene_density=0.18),
            VisionWindow(start=20.0, end=30.0, score=0.85, motion=7.6, saturation=30.0, scene_density=0.16),
            VisionWindow(start=30.0, end=40.0, score=0.80, motion=7.4, saturation=30.0, scene_density=0.14),
        ]
        candidates = _rank_vision_candidates(
            windows,
            min_gap_seconds=8.0,
            clip_before=12.0,
            clip_after=18.0,
        )
        self.assertGreaterEqual(len(candidates), 3)

    def test_build_segments_prioritizes_detected_death_cues(self) -> None:
        windows = [
            VisionWindow(start=100.0, end=120.0, score=0.75, motion=9.0, saturation=46.0, scene_density=0.25),
            VisionWindow(start=120.0, end=140.0, score=0.70, motion=8.2, saturation=45.0, scene_density=0.2),
            VisionWindow(start=140.0, end=160.0, score=0.22, motion=2.2, saturation=9.0, scene_density=0.02),
            VisionWindow(start=160.0, end=180.0, score=0.18, motion=1.9, saturation=8.0, scene_density=0.02),
            VisionWindow(start=220.0, end=240.0, score=0.60, motion=6.8, saturation=40.0, scene_density=0.15),
            VisionWindow(start=240.0, end=260.0, score=0.58, motion=6.5, saturation=38.0, scene_density=0.15),
        ]
        segments, _ = build_segments(
            [130.0, 235.0],
            duration_seconds=500.0,
            clip_before=10.0,
            clip_after=12.0,
            min_gap_seconds=20.0,
            max_clips=2,
            target_duration_seconds=180.0,
            intro_seconds=0.0,
            outro_seconds=30.0,
            vision_windows=windows,
        )
        self.assertTrue(any(segment.start <= 135.0 <= segment.end for segment in segments))

    def test_build_segments_samples_forced_cues_across_timeline(self) -> None:
        with (
            patch(
                "league_video_editor.cli._detect_death_cues",
                return_value=[100.0, 180.0, 260.0, 340.0, 420.0, 560.0],
            ),
            patch("league_video_editor.cli._detect_combat_cues", return_value=[]),
        ):
            segments, _ = build_segments(
                [],
                duration_seconds=640.0,
                clip_before=8.0,
                clip_after=12.0,
                min_gap_seconds=20.0,
                max_clips=3,
                target_duration_seconds=120.0,
                intro_seconds=0.0,
                outro_seconds=0.0,
                vision_windows=[],
            )
        self.assertEqual(len(segments), 3)
        self.assertTrue(any(segment.start <= 100.0 <= segment.end for segment in segments))
        self.assertTrue(any(segment.start <= 560.0 <= segment.end for segment in segments))

    def test_build_segments_prioritizes_ai_forced_cues(self) -> None:
        segments, _ = build_segments(
            [50.0, 500.0],
            duration_seconds=700.0,
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=20.0,
            max_clips=2,
            target_duration_seconds=140.0,
            intro_seconds=0.0,
            outro_seconds=0.0,
            vision_windows=[],
            ai_priority_cues=[210.0, 510.0],
        )
        self.assertEqual(len(segments), 2)
        self.assertTrue(any(segment.start <= 210.0 <= segment.end for segment in segments))
        self.assertTrue(any(segment.start <= 510.0 <= segment.end for segment in segments))

    def test_score_transcript_text_rewards_high_impact_lol_callouts(self) -> None:
        high_score, high_hits = _score_transcript_text("PENTAKILL! We got Baron and end now!")
        low_score, low_hits = _score_transcript_text("just clearing wave and farming top lane")
        self.assertGreater(high_score, low_score)
        self.assertIn("pentakill", high_hits)
        self.assertIn("baron", high_hits)
        self.assertEqual(low_hits, ())

    def test_score_transcript_text_ignores_placeholder_tokens(self) -> None:
        score, hits = _score_transcript_text("[BLANK_AUDIO] [MUSIC]")
        self.assertEqual(score, 0.0)
        self.assertEqual(hits, ())

    def test_clean_moment_blurb_rejects_noisy_ocr_snippets(self) -> None:
        noisy = "Te le es | te Ow2 40/00 847 0607 * a Sea 5:120 26m"
        self.assertIsNone(_clean_moment_blurb(noisy))

    def test_score_transcript_text_does_not_match_partial_word_keywords(self) -> None:
        score, hits = _score_transcript_text("peaceful farming and safe lane reset")
        self.assertEqual(score, 0.0)
        self.assertEqual(hits, ())

    def test_parse_whisper_json_cues_extracts_segments_and_thresholds(self) -> None:
        with TemporaryDirectory() as temp_dir:
            transcript_path = Path(temp_dir) / "transcript.json"
            transcript_path.write_text(
                json.dumps(
                    {
                        "transcription": [
                            {
                                "timestamps": {"from": "00:03:00,000", "to": "00:03:04,500"},
                                "offsets": {"from": 180000, "to": 184500},
                                "text": "Shutdown! Huge teamfight at dragon.",
                            },
                            {
                                "timestamps": {"from": "00:05:00,000", "to": "00:05:02,000"},
                                "offsets": {"from": 300000, "to": 302000},
                                "text": "just walking to lane",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            cues = _parse_whisper_json_cues(transcript_path, cue_threshold=0.5)

        self.assertEqual(len(cues), 1)
        self.assertAlmostEqual(cues[0].start, 180.0)
        self.assertAlmostEqual(cues[0].end, 184.5)
        self.assertIn("shutdown", cues[0].keywords)
        self.assertIn("dragon", cues[0].keywords)

    def test_parse_whisper_json_cues_ignores_placeholder_segments(self) -> None:
        with TemporaryDirectory() as temp_dir:
            transcript_path = Path(temp_dir) / "transcript.json"
            transcript_path.write_text(
                json.dumps(
                    {
                        "transcription": [
                            {
                                "offsets": {"from": 1000, "to": 3000},
                                "text": "[BLANK_AUDIO]",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            cues = _parse_whisper_json_cues(transcript_path, cue_threshold=0.12)

        self.assertEqual(cues, [])

    def test_collect_local_ai_cues_extracts_audio_and_parses_json(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mov"
            input_path.touch()
            model_path = Path(temp_dir) / "model.bin"
            model_path.touch()

            def fake_run_command(cmd: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
                if cmd and cmd[0] == "whisper-cli":
                    output_prefix = cmd[cmd.index("-of") + 1]
                    Path(f"{output_prefix}.json").write_text(
                        json.dumps(
                            {
                                "transcription": [
                                    {
                                        "offsets": {"from": 1000, "to": 4000},
                                        "text": "Shutdown into Baron!",
                                    }
                                ]
                            }
                        ),
                        encoding="utf-8",
                    )
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

            with (
                patch("league_video_editor.cli._resolve_whisper_cpp_binary", return_value="whisper-cli"),
                patch("league_video_editor.cli._extract_audio_for_whisper") as extract_audio,
                patch("league_video_editor.cli._run_command", side_effect=fake_run_command),
            ):
                cues = _collect_local_ai_cues(
                    input_path=input_path,
                    whisper_model=model_path,
                    whisper_binary="whisper-cli",
                    whisper_language="en",
                    whisper_threads=2,
                    cue_threshold=0.2,
                )

        self.assertEqual(extract_audio.call_count, 1)
        self.assertEqual(len(cues), 1)
        self.assertAlmostEqual(cues[0].start, 1.0)
        self.assertIn("shutdown", cues[0].keywords)

    def test_collect_local_ai_cues_wraps_audio_extract_errors(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mov"
            input_path.touch()
            model_path = Path(temp_dir) / "model.bin"
            model_path.touch()
            with (
                patch("league_video_editor.cli._resolve_whisper_cpp_binary", return_value="whisper-cli"),
                patch(
                    "league_video_editor.cli._extract_audio_for_whisper",
                    side_effect=subprocess.CalledProcessError(
                        returncode=1,
                        cmd=["ffmpeg"],
                        stderr="No audio stream",
                    ),
                ),
            ):
                with self.assertRaisesRegex(EditorError, "Could not extract audio"):
                    _collect_local_ai_cues(
                        input_path=input_path,
                        whisper_model=model_path,
                        whisper_binary="whisper-cli",
                        whisper_language="en",
                        whisper_threads=2,
                        cue_threshold=0.2,
                    )

    def test_collect_local_ai_cues_can_disable_vad(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mov"
            input_path.touch()
            model_path = Path(temp_dir) / "model.bin"
            model_path.touch()
            captured_whisper_cmds: list[list[str]] = []

            def fake_run_command(cmd: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
                if cmd and cmd[0] == "whisper-cli":
                    captured_whisper_cmds.append(cmd)
                    output_prefix = cmd[cmd.index("-of") + 1]
                    Path(f"{output_prefix}.json").write_text(
                        json.dumps({"transcription": []}),
                        encoding="utf-8",
                    )
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

            with (
                patch("league_video_editor.cli._resolve_whisper_cpp_binary", return_value="whisper-cli"),
                patch("league_video_editor.cli._extract_audio_for_whisper"),
                patch("league_video_editor.cli._run_command", side_effect=fake_run_command),
            ):
                _collect_local_ai_cues(
                    input_path=input_path,
                    whisper_model=model_path,
                    whisper_binary="whisper-cli",
                    whisper_language="en",
                    whisper_threads=2,
                    cue_threshold=0.2,
                    whisper_vad=False,
                )

        self.assertEqual(len(captured_whisper_cmds), 1)
        self.assertNotIn("--vad", captured_whisper_cmds[0])

    def test_collect_ocr_cues_extracts_keyword_frames(self) -> None:
        with (
            patch("league_video_editor.cli._resolve_tesseract_binary", return_value="tesseract"),
            patch(
                "league_video_editor.cli._extract_ocr_frames",
                return_value=[Path("f1.jpg"), Path("f2.jpg"), Path("f3.jpg")],
            ),
            patch(
                "league_video_editor.cli._run_tesseract_ocr",
                side_effect=[
                    "",
                    "Enemy has slain Baron Nashor!",
                    "just farming side lane",
                ],
            ),
        ):
            cues = _collect_ocr_cues(
                input_path=Path("input.mov"),
                tesseract_binary="tesseract",
                sample_fps=0.25,
                cue_threshold=0.16,
            )

        self.assertEqual(len(cues), 1)
        self.assertTrue(any("baron" in hit for hit in cues[0].keywords))

    def test_extract_audio_for_whisper_supports_explicit_stream_mapping(self) -> None:
        with patch("league_video_editor.cli._run_command") as run_command:
            run_command.return_value = subprocess.CompletedProcess(
                args=["ffmpeg"], returncode=0, stdout="", stderr=""
            )
            _extract_audio_for_whisper(
                input_path=Path("input.mov"),
                output_audio_path=Path("out.wav"),
                audio_stream_index=2,
            )

        called_cmd = run_command.call_args.args[0]
        self.assertIn("-map", called_cmd)
        self.assertIn("0:a:2", called_cmd)

    def test_boost_vision_windows_with_ai_cues_increases_nearby_scores(self) -> None:
        windows = [
            VisionWindow(start=0.0, end=20.0, score=0.30, motion=3.0, saturation=22.0, scene_density=0.05),
            VisionWindow(start=20.0, end=40.0, score=0.30, motion=3.2, saturation=23.0, scene_density=0.05),
        ]
        ai_cues = [
            TranscriptionCue(
                start=18.0,
                end=22.0,
                score=0.90,
                text="Ace and Baron now!",
                keywords=("ace", "baron"),
            )
        ]
        boosted = _boost_vision_windows_with_ai_cues(windows, ai_cues, radius_seconds=16.0)
        self.assertEqual(len(boosted), 2)
        self.assertGreater(boosted[1].score, windows[1].score)
        self.assertGreater(boosted[0].score, windows[0].score)

    def test_build_watchability_report_outputs_score_and_recommendations(self) -> None:
        vision_windows = [
            VisionWindow(start=0.0, end=12.0, score=0.15, motion=1.2, saturation=10.0, scene_density=0.01),
            VisionWindow(start=12.0, end=24.0, score=0.25, motion=2.0, saturation=12.0, scene_density=0.03),
            VisionWindow(start=24.0, end=36.0, score=0.65, motion=8.5, saturation=35.0, scene_density=0.18),
            VisionWindow(start=36.0, end=48.0, score=0.80, motion=10.2, saturation=40.0, scene_density=0.22),
            VisionWindow(start=48.0, end=60.0, score=0.20, motion=1.5, saturation=9.0, scene_density=0.02),
        ]
        report = _build_watchability_report(
            duration_seconds=60.0,
            events=[8.0, 22.0, 37.0],
            vision_windows=vision_windows,
            scene_threshold_used=0.35,
        )
        self.assertIn("watchability_score", report)
        self.assertIn("rating", report)
        self.assertIn("highlight_quality_score", report)
        self.assertIn("quality_rating", report)
        self.assertIn("youtube_score", report)
        self.assertIn("score_blend", report)
        self.assertIn("scene_threshold_used", report)
        self.assertIn("recommendations", report)
        self.assertGreaterEqual(float(report["watchability_score"]), 0.0)
        self.assertLessEqual(float(report["watchability_score"]), 100.0)
        self.assertGreaterEqual(float(report["highlight_quality_score"]), 0.0)
        self.assertLessEqual(float(report["highlight_quality_score"]), 100.0)
        self.assertGreaterEqual(float(report["youtube_score"]), 0.0)
        self.assertLessEqual(float(report["youtube_score"]), 100.0)

    def test_analyze_watchability_emits_overall_loading_bar(self) -> None:
        vision_windows = [
            VisionWindow(start=0.0, end=12.0, score=0.6, motion=8.0, saturation=30.0, scene_density=0.1),
            VisionWindow(start=12.0, end=24.0, score=0.7, motion=9.0, saturation=32.0, scene_density=0.12),
        ]
        with (
            patch("league_video_editor.cli._probe_duration_seconds", return_value=24.0),
            patch(
                "league_video_editor.cli.detect_scene_events_adaptive",
                return_value=([4.0, 10.0], 0.35),
            ) as detect_scene_events,
            patch("league_video_editor.cli.score_vision_activity", return_value=vision_windows) as score_vision_activity,
            patch("league_video_editor.cli.print") as mocked_print,
        ):
            report = analyze_watchability(
                input_path=Path("input.mp4"),
                scene_threshold=0.35,
                vision_sample_fps=1.0,
                vision_window_seconds=12.0,
                vision_step_seconds=6.0,
            )

        self.assertIn("watchability_score", report)
        self.assertEqual(detect_scene_events.call_count, 1)
        self.assertEqual(score_vision_activity.call_count, 1)

        detect_kwargs = detect_scene_events.call_args.kwargs
        self.assertTrue(detect_kwargs["show_progress"])
        self.assertIsNone(detect_kwargs["progress_label"])
        self.assertIsNotNone(detect_kwargs["progress_callback"])

        score_kwargs = score_vision_activity.call_args.kwargs
        self.assertTrue(score_kwargs["show_progress"])
        self.assertIsNone(score_kwargs["progress_label"])
        self.assertIsNotNone(score_kwargs["progress_callback"])

        printed_lines = [str(call.args[0]) for call in mocked_print.call_args_list if call.args]
        self.assertTrue(any("Analyzing watchability" in line for line in printed_lines))
        self.assertTrue(any("100.0%" in line for line in printed_lines))

    def test_analyze_watchability_can_disable_loading_bar(self) -> None:
        vision_windows = [
            VisionWindow(start=0.0, end=12.0, score=0.6, motion=8.0, saturation=30.0, scene_density=0.1),
        ]
        with (
            patch("league_video_editor.cli._probe_duration_seconds", return_value=12.0),
            patch(
                "league_video_editor.cli.detect_scene_events_adaptive",
                return_value=([4.0], 0.35),
            ) as detect_scene_events,
            patch("league_video_editor.cli.score_vision_activity", return_value=vision_windows) as score_vision_activity,
            patch("league_video_editor.cli.print") as mocked_print,
        ):
            analyze_watchability(
                input_path=Path("input.mp4"),
                scene_threshold=0.35,
                vision_sample_fps=1.0,
                vision_window_seconds=12.0,
                vision_step_seconds=6.0,
                show_progress=False,
            )

        detect_kwargs = detect_scene_events.call_args.kwargs
        self.assertFalse(detect_kwargs["show_progress"])
        self.assertEqual(detect_kwargs["progress_label"], "Analyzing scenes")
        self.assertIsNone(detect_kwargs["progress_callback"])

        score_kwargs = score_vision_activity.call_args.kwargs
        self.assertFalse(score_kwargs["show_progress"])
        self.assertEqual(score_kwargs["progress_label"], "Scoring gameplay")
        self.assertIsNone(score_kwargs["progress_callback"])

        self.assertFalse(mocked_print.called)

    def test_build_segments_with_zero_max_clips_returns_empty(self) -> None:
        segments, used_fallback = build_segments(
            [10.0, 20.0, 30.0],
            duration_seconds=1200.0,
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=18.0,
            max_clips=0,
        )
        self.assertEqual(segments, [])
        self.assertFalse(used_fallback)

    def test_read_plan_missing_file_raises_editor_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            missing_plan = Path(temp_dir) / "missing-plan.json"
            with self.assertRaisesRegex(EditorError, "Could not read plan file"):
                read_plan(missing_plan)

    def test_read_plan_invalid_json_raises_editor_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            plan_path = Path(temp_dir) / "plan.json"
            plan_path.write_text("{not-json", encoding="utf-8")
            with self.assertRaisesRegex(EditorError, "not valid JSON"):
                read_plan(plan_path)

    def test_read_plan_invalid_segment_shape_raises_editor_error(self) -> None:
        with TemporaryDirectory() as temp_dir:
            plan_path = Path(temp_dir) / "plan.json"
            plan_path.write_text(
                json.dumps({"segments": [{"start": "oops", "end": 10.0}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EditorError, "Segment at index 0"):
                read_plan(plan_path)

    def test_read_plan_rejects_non_finite_segment_values(self) -> None:
        with TemporaryDirectory() as temp_dir:
            plan_path = Path(temp_dir) / "plan.json"
            plan_path.write_text(
                json.dumps({"segments": [{"start": float("nan"), "end": 10.0}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EditorError, "non-finite"):
                read_plan(plan_path)

    def test_read_plan_rejects_non_positive_duration_segment(self) -> None:
        with TemporaryDirectory() as temp_dir:
            plan_path = Path(temp_dir) / "plan.json"
            plan_path.write_text(
                json.dumps({"segments": [{"start": 10.0, "end": 10.0}]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EditorError, "end > start"):
                read_plan(plan_path)

    def test_write_plan_rejects_non_finite_values(self) -> None:
        with TemporaryDirectory() as temp_dir:
            plan_path = Path(temp_dir) / "plan.json"
            with self.assertRaisesRegex(EditorError, "non-finite"):
                write_plan(
                    input_path=Path("input.mp4"),
                    output_path=plan_path,
                    duration_seconds=120.0,
                    events=[],
                    segments=[],
                    used_fallback=False,
                    settings={
                        "scene_threshold": float("nan"),
                        "clip_before": 8.0,
                        "clip_after": 12.0,
                        "min_gap_seconds": 18.0,
                        "max_clips": 20,
                    },
                )

    def test_probe_duration_rejects_non_numeric_output(self) -> None:
        with patch(
            "league_video_editor.cli._run_command",
            return_value=subprocess.CompletedProcess(
                args=["ffprobe"], returncode=0, stdout="N/A\n", stderr=""
            ),
        ):
            with self.assertRaisesRegex(EditorError, "Could not parse duration"):
                _probe_duration_seconds(Path("input.mp4"))

    def test_probe_duration_rejects_non_finite_output(self) -> None:
        with patch(
            "league_video_editor.cli._run_command",
            return_value=subprocess.CompletedProcess(
                args=["ffprobe"], returncode=0, stdout="inf\n", stderr=""
            ),
        ):
            with self.assertRaisesRegex(EditorError, "valid finite value"):
                _probe_duration_seconds(Path("input.mp4"))

    def test_detect_watchability_crop_filter_uses_most_common_crop(self) -> None:
        crop_output = "\n".join(
            [
                "crop=1920:1080:0:0",
                "crop=1920:1080:0:0",
                "crop=1880:1040:20:20",
            ]
        )
        with patch(
            "league_video_editor.cli._run_command",
            return_value=subprocess.CompletedProcess(
                args=["ffmpeg"], returncode=0, stdout="", stderr=crop_output
            ),
        ):
            crop_filter = _detect_watchability_crop_filter(
                Path("input.mp4"),
                duration_seconds=120.0,
            )
        self.assertEqual(crop_filter, "crop=1920:1080:0:0")

    def test_validate_cli_options_rejects_non_finite_scene_threshold(self) -> None:
        args = argparse.Namespace(
            command="analyze",
            scene_threshold=float("nan"),
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=18.0,
            max_clips=20,
            target_duration_seconds=600.0,
            target_duration_ratio=0.6,
            intro_seconds=45.0,
            outro_seconds=60.0,
        )
        with self.assertRaisesRegex(EditorError, "finite number"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_invalid_ranges(self) -> None:
        args = argparse.Namespace(
            command="auto",
            scene_threshold=1.1,
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=18.0,
            max_clips=0,
            target_duration_seconds=600.0,
            target_duration_ratio=0.6,
            intro_seconds=45.0,
            outro_seconds=60.0,
            vision_scoring="heuristic",
            vision_sample_fps=1.0,
            vision_window_seconds=12.0,
            vision_step_seconds=6.0,
            crf=20,
            crossfade_seconds=0.0,
            audio_fade_seconds=0.03,
            video_encoder="libx264",
            preset="medium",
        )
        with self.assertRaisesRegex(EditorError, "scene-threshold must be between 0.0 and 1.0"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_invalid_vision_sampling_options(self) -> None:
        args = argparse.Namespace(
            command="analyze",
            scene_threshold=0.35,
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=18.0,
            max_clips=20,
            target_duration_seconds=600.0,
            target_duration_ratio=0.6,
            intro_seconds=45.0,
            outro_seconds=60.0,
            vision_scoring="heuristic",
            vision_sample_fps=0.0,
            vision_window_seconds=12.0,
            vision_step_seconds=6.0,
        )
        with self.assertRaisesRegex(EditorError, "vision-sample-fps"):
            _validate_cli_options(args)

    def test_validate_cli_options_requires_whisper_model_for_local_ai(self) -> None:
        args = argparse.Namespace(
            command="analyze",
            scene_threshold=0.35,
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=18.0,
            max_clips=20,
            target_duration_seconds=600.0,
            target_duration_ratio=0.6,
            intro_seconds=45.0,
            outro_seconds=60.0,
            vision_scoring="local-ai",
            vision_sample_fps=1.0,
            vision_window_seconds=12.0,
            vision_step_seconds=6.0,
            whisper_model=None,
            whisper_bin="auto",
            whisper_language="en",
            whisper_threads=4,
            ai_cue_threshold=0.55,
        )
        with self.assertRaisesRegex(EditorError, "whisper-model is required"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_invalid_ai_threshold(self) -> None:
        args = argparse.Namespace(
            command="analyze",
            scene_threshold=0.35,
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=18.0,
            max_clips=20,
            target_duration_seconds=600.0,
            target_duration_ratio=0.6,
            intro_seconds=45.0,
            outro_seconds=60.0,
            vision_scoring="heuristic",
            vision_sample_fps=1.0,
            vision_window_seconds=12.0,
            vision_step_seconds=6.0,
            whisper_model=None,
            whisper_bin="auto",
            whisper_language="en",
            whisper_threads=4,
            ai_cue_threshold=1.2,
        )
        with self.assertRaisesRegex(EditorError, "ai-cue-threshold"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_invalid_whisper_audio_stream(self) -> None:
        args = argparse.Namespace(
            command="analyze",
            scene_threshold=0.35,
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=18.0,
            max_clips=20,
            target_duration_seconds=600.0,
            target_duration_ratio=0.6,
            intro_seconds=45.0,
            outro_seconds=60.0,
            vision_scoring="heuristic",
            vision_sample_fps=1.0,
            vision_window_seconds=12.0,
            vision_step_seconds=6.0,
            whisper_model=None,
            whisper_bin="auto",
            whisper_language="en",
            whisper_threads=4,
            whisper_audio_stream=-2,
            ai_cue_threshold=0.4,
        )
        with self.assertRaisesRegex(EditorError, "whisper-audio-stream"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_invalid_whisper_vad_threshold(self) -> None:
        args = argparse.Namespace(
            command="analyze",
            scene_threshold=0.35,
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=18.0,
            max_clips=20,
            target_duration_seconds=600.0,
            target_duration_ratio=0.6,
            intro_seconds=45.0,
            outro_seconds=60.0,
            vision_scoring="heuristic",
            vision_sample_fps=1.0,
            vision_window_seconds=12.0,
            vision_step_seconds=6.0,
            whisper_vad_threshold=1.3,
        )
        with self.assertRaisesRegex(EditorError, "whisper-vad-threshold"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_invalid_ocr_scoring_mode(self) -> None:
        args = argparse.Namespace(
            command="analyze",
            scene_threshold=0.35,
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=18.0,
            max_clips=20,
            target_duration_seconds=600.0,
            target_duration_ratio=0.6,
            intro_seconds=45.0,
            outro_seconds=60.0,
            vision_scoring="heuristic",
            vision_sample_fps=1.0,
            vision_window_seconds=12.0,
            vision_step_seconds=6.0,
            ocr_cue_scoring="bad-mode",
        )
        with self.assertRaisesRegex(EditorError, "ocr-cue-scoring"):
            _validate_cli_options(args)

    def test_build_arg_parser_uses_higher_default_max_clips(self) -> None:
        parser = build_arg_parser()
        analyze_args = parser.parse_args(["analyze", "input.mp4"])
        auto_args = parser.parse_args(["auto", "input.mp4"])
        thumbnail_args = parser.parse_args(["thumbnail", "input.mp4"])
        description_args = parser.parse_args(["description", "input.mp4"])
        self.assertEqual(analyze_args.max_clips, 24)
        self.assertEqual(auto_args.max_clips, 24)
        self.assertTrue(analyze_args.end_on_result)
        self.assertTrue(analyze_args.one_shot_smart)
        self.assertTrue(auto_args.end_on_result)
        self.assertTrue(auto_args.one_shot_smart)
        self.assertFalse(auto_args.auto_optimize)
        self.assertEqual(auto_args.optimize_metric, "youtube")
        self.assertEqual(thumbnail_args.output, Path("thumbnail.jpg"))
        self.assertIsNone(thumbnail_args.timestamp)
        self.assertEqual(thumbnail_args.width, 1280)
        self.assertEqual(thumbnail_args.height, 720)
        self.assertAlmostEqual(thumbnail_args.champion_scale, 0.55, places=2)
        self.assertEqual(thumbnail_args.champion_anchor, "right")
        self.assertIsNone(thumbnail_args.champion_overlay)
        self.assertIsNone(thumbnail_args.headline)
        self.assertEqual(thumbnail_args.headline_size, 118)
        self.assertEqual(thumbnail_args.headline_color, "white")
        self.assertTrue(thumbnail_args.auto_crop)
        self.assertTrue(thumbnail_args.enhance)
        self.assertEqual(description_args.output, Path("youtube-description.txt"))
        self.assertEqual(description_args.scene_threshold, 0.20)
        self.assertEqual(description_args.title_count, 3)
        self.assertEqual(description_args.max_moments, 8)
        self.assertEqual(description_args.ocr_cue_scoring, "auto")

    def test_validate_cli_options_rejects_negative_target_duration(self) -> None:
        args = argparse.Namespace(
            command="analyze",
            scene_threshold=0.35,
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=18.0,
            max_clips=20,
            target_duration_seconds=-1.0,
            target_duration_ratio=0.6,
            intro_seconds=45.0,
            outro_seconds=60.0,
            vision_scoring="heuristic",
            vision_sample_fps=1.0,
            vision_window_seconds=12.0,
            vision_step_seconds=6.0,
        )
        with self.assertRaisesRegex(EditorError, "target-duration-seconds"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_invalid_target_duration_ratio(self) -> None:
        args = argparse.Namespace(
            command="analyze",
            scene_threshold=0.35,
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=18.0,
            max_clips=20,
            target_duration_seconds=0.0,
            target_duration_ratio=1.2,
            intro_seconds=45.0,
            outro_seconds=60.0,
            vision_scoring="heuristic",
            vision_sample_fps=1.0,
            vision_window_seconds=12.0,
            vision_step_seconds=6.0,
        )
        with self.assertRaisesRegex(EditorError, "target-duration-ratio"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_invalid_watchability_options(self) -> None:
        args = argparse.Namespace(
            command="watchability",
            scene_threshold=0.35,
            vision_sample_fps=1.0,
            vision_window_seconds=0.0,
            vision_step_seconds=6.0,
        )
        with self.assertRaisesRegex(EditorError, "vision-window-seconds"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_invalid_thumbnail_options(self) -> None:
        args = argparse.Namespace(
            command="thumbnail",
            scene_threshold=0.20,
            vision_sample_fps=0.75,
            vision_window_seconds=8.0,
            vision_step_seconds=4.0,
            timestamp=-1.0,
            width=1280,
            height=720,
            quality=2,
            champion_scale=0.55,
            champion_anchor="right",
            champion_overlay=None,
            headline=None,
            headline_size=118,
            headline_color="white",
            headline_font=None,
            headline_y_ratio=0.06,
            output=Path("thumbnail.jpg"),
        )
        with self.assertRaisesRegex(EditorError, "timestamp must be >= 0"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_invalid_thumbnail_extension(self) -> None:
        args = argparse.Namespace(
            command="thumbnail",
            scene_threshold=0.20,
            vision_sample_fps=0.75,
            vision_window_seconds=8.0,
            vision_step_seconds=4.0,
            timestamp=None,
            width=1280,
            height=720,
            quality=2,
            champion_scale=0.55,
            champion_anchor="right",
            champion_overlay=None,
            headline=None,
            headline_size=118,
            headline_color="white",
            headline_font=None,
            headline_y_ratio=0.06,
            output=Path("thumbnail.gif"),
        )
        with self.assertRaisesRegex(EditorError, "must end with"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_missing_champion_overlay_file(self) -> None:
        args = argparse.Namespace(
            command="thumbnail",
            scene_threshold=0.20,
            vision_sample_fps=0.75,
            vision_window_seconds=8.0,
            vision_step_seconds=4.0,
            timestamp=None,
            width=1280,
            height=720,
            quality=2,
            champion_scale=0.55,
            champion_anchor="right",
            champion_overlay=Path("/tmp/not-there-champ.png"),
            headline=None,
            headline_size=118,
            headline_color="white",
            headline_font=None,
            headline_y_ratio=0.06,
            output=Path("thumbnail.jpg"),
        )
        with self.assertRaisesRegex(EditorError, "Champion overlay file not found"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_invalid_description_options(self) -> None:
        args = argparse.Namespace(
            command="description",
            scene_threshold=0.20,
            vision_sample_fps=1.0,
            vision_window_seconds=12.0,
            vision_step_seconds=6.0,
            whisper_model=None,
            whisper_bin="auto",
            whisper_language="en",
            whisper_threads=4,
            whisper_audio_stream=-1,
            whisper_vad=True,
            whisper_vad_threshold=0.50,
            whisper_vad_model=None,
            ocr_cue_scoring="auto",
            tesseract_bin="auto",
            ocr_sample_fps=0.25,
            ocr_cue_threshold=0.16,
            ai_cue_threshold=0.40,
            max_moments=0,
            title_count=3,
        )
        with self.assertRaisesRegex(EditorError, "max-moments"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_crf_out_of_range(self) -> None:
        args = argparse.Namespace(command="full", crf=52, video_encoder="libx264", preset="medium")
        with self.assertRaisesRegex(EditorError, "crf must be between 0 and 51"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_negative_crossfade(self) -> None:
        args = argparse.Namespace(
            command="render",
            crf=20,
            crossfade_seconds=-0.1,
            audio_fade_seconds=0.03,
            video_encoder="libx264",
            preset="medium",
        )
        with self.assertRaisesRegex(EditorError, "crossfade-seconds"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_invalid_video_encoder(self) -> None:
        args = argparse.Namespace(
            command="full",
            crf=20,
            video_encoder="not-real",
            preset="medium",
        )
        with self.assertRaisesRegex(EditorError, "video-encoder must be one of"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_non_default_preset_for_hardware_encoder(self) -> None:
        args = argparse.Namespace(
            command="full",
            crf=20,
            video_encoder="h264_videotoolbox",
            preset="fast",
        )
        with self.assertRaisesRegex(EditorError, "preset applies only to libx264"):
            _validate_cli_options(args)

    def test_render_highlights_creates_output_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            input_path.touch()
            output_path = Path(temp_dir) / "nested" / "rendered.mp4"

            with (
                patch("league_video_editor.cli.read_plan", return_value=[Segment(0.0, 1.0)]),
                patch("league_video_editor.cli._has_audio_stream", return_value=False),
                patch("league_video_editor.cli._run_command") as run_command,
            ):
                run_command.return_value = subprocess.CompletedProcess(
                    args=["ffmpeg"], returncode=0, stdout="", stderr=""
                )
                render_highlights(
                    input_path=input_path,
                    plan_path=Path(temp_dir) / "plan.json",
                    output_path=output_path,
                    crf=20,
                    preset="medium",
                    video_encoder="libx264",
                    allow_upscale=False,
                    crossfade_seconds=0.0,
                    audio_fade_seconds=0.03,
                    two_pass_loudnorm=False,
                )

            self.assertTrue(output_path.parent.is_dir())

    def test_transcode_full_match_creates_output_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            input_path.touch()
            output_path = Path(temp_dir) / "nested" / "full.mp4"

            with (
                patch("league_video_editor.cli._has_audio_stream", return_value=False),
                patch("league_video_editor.cli._run_command") as run_command,
            ):
                run_command.return_value = subprocess.CompletedProcess(
                    args=["ffmpeg"], returncode=0, stdout="", stderr=""
                )
                transcode_full_match(
                    input_path=input_path,
                    output_path=output_path,
                    crf=20,
                    preset="medium",
                    video_encoder="libx264",
                    allow_upscale=False,
                    two_pass_loudnorm=False,
                )

            self.assertTrue(output_path.parent.is_dir())

    def test_generate_thumbnail_uses_manual_timestamp(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "thumb.jpg"
            input_path.touch()
            with (
                patch("league_video_editor.cli._probe_duration_seconds", return_value=120.0),
                patch(
                    "league_video_editor.cli._detect_watchability_crop_filter",
                    return_value="crop=1920:1080:0:0",
                ),
                patch("league_video_editor.cli._run_command") as run_command,
            ):
                run_command.return_value = subprocess.CompletedProcess(
                    args=["ffmpeg"], returncode=0, stdout="", stderr=""
                )
                result = generate_thumbnail(
                    input_path=input_path,
                    output_path=output_path,
                    timestamp_seconds=42.5,
                    scene_threshold=0.20,
                    vision_sample_fps=0.75,
                    vision_window_seconds=8.0,
                    vision_step_seconds=4.0,
                    width=1280,
                    height=720,
                    quality=2,
                    auto_crop=True,
                    enhance=True,
                    champion_overlay_path=None,
                    champion_scale=0.55,
                    champion_anchor="right",
                    headline_text=None,
                    headline_size=118,
                    headline_color="white",
                    headline_font=None,
                    headline_y_ratio=0.06,
                )

            self.assertFalse(bool(result["auto_selected"]))
            self.assertAlmostEqual(float(result["timestamp_seconds"]), 42.5, delta=0.01)
            called_cmd = run_command.call_args.args[0]
            self.assertIn("-ss", called_cmd)
            self.assertIn("42.500", called_cmd)
            self.assertIn("-q:v", called_cmd)
            self.assertTrue(output_path.parent.is_dir())

    def test_generate_thumbnail_auto_selects_from_vision_windows(self) -> None:
        windows = [
            VisionWindow(start=12.0, end=20.0, score=0.35, motion=3.0, saturation=16.0, scene_density=0.05),
            VisionWindow(start=46.0, end=54.0, score=0.92, motion=10.0, saturation=42.0, scene_density=0.20),
            VisionWindow(start=70.0, end=78.0, score=0.40, motion=4.0, saturation=18.0, scene_density=0.06),
        ]
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "thumb.png"
            input_path.touch()
            with (
                patch("league_video_editor.cli._probe_duration_seconds", return_value=100.0),
                patch(
                    "league_video_editor.cli.detect_scene_events_adaptive",
                    return_value=([10.0, 32.0, 50.0], 0.20),
                ) as detect_events,
                patch("league_video_editor.cli.score_vision_activity", return_value=windows) as score_vision,
                patch("league_video_editor.cli._run_command") as run_command,
            ):
                run_command.return_value = subprocess.CompletedProcess(
                    args=["ffmpeg"], returncode=0, stdout="", stderr=""
                )
                result = generate_thumbnail(
                    input_path=input_path,
                    output_path=output_path,
                    timestamp_seconds=None,
                    scene_threshold=0.20,
                    vision_sample_fps=0.75,
                    vision_window_seconds=8.0,
                    vision_step_seconds=4.0,
                    width=1280,
                    height=720,
                    quality=2,
                    auto_crop=False,
                    enhance=False,
                    champion_overlay_path=None,
                    champion_scale=0.55,
                    champion_anchor="right",
                    headline_text=None,
                    headline_size=118,
                    headline_color="white",
                    headline_font=None,
                    headline_y_ratio=0.06,
                )

            self.assertTrue(bool(result["auto_selected"]))
            self.assertGreater(float(result["timestamp_seconds"]), 40.0)
            self.assertLess(float(result["timestamp_seconds"]), 60.0)
            self.assertEqual(detect_events.call_count, 1)
            self.assertEqual(score_vision.call_count, 1)
            called_cmd = run_command.call_args.args[0]
            self.assertIn(str(output_path), called_cmd)
            self.assertNotIn("-q:v", called_cmd)

    def test_generate_thumbnail_applies_champion_overlay_and_headline(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            champion_png = Path(temp_dir) / "champ.png"
            output_path = Path(temp_dir) / "thumb.jpg"
            input_path.touch()
            champion_png.touch()
            with (
                patch("league_video_editor.cli._probe_duration_seconds", return_value=180.0),
                patch("league_video_editor.cli._create_headline_overlay_image"),
                patch("league_video_editor.cli._run_command") as run_command,
            ):
                run_command.return_value = subprocess.CompletedProcess(
                    args=["ffmpeg"], returncode=0, stdout="", stderr=""
                )
                result = generate_thumbnail(
                    input_path=input_path,
                    output_path=output_path,
                    timestamp_seconds=50.0,
                    scene_threshold=0.20,
                    vision_sample_fps=0.75,
                    vision_window_seconds=8.0,
                    vision_step_seconds=4.0,
                    width=1280,
                    height=720,
                    quality=2,
                    auto_crop=False,
                    enhance=True,
                    champion_overlay_path=champion_png,
                    champion_scale=0.60,
                    champion_anchor="left",
                    headline_text="KATARINA CARRY",
                    headline_size=116,
                    headline_color="#ffd400",
                    headline_font=None,
                    headline_y_ratio=0.08,
                )

            self.assertTrue(bool(result["used_champion_overlay"]))
            self.assertTrue(bool(result["used_headline_text"]))
            called_cmd = run_command.call_args.args[0]
            self.assertIn(str(champion_png), called_cmd)
            self.assertIn("-filter_complex", called_cmd)
            filter_graph = called_cmd[called_cmd.index("-filter_complex") + 1]
            self.assertIn("overlay=", filter_graph)
            self.assertIn("[2:v]format=rgba[text]", filter_graph)
            self.assertGreaterEqual(filter_graph.count("overlay="), 2)

    def test_generate_youtube_description_writes_ctr_package(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "youtube-description.txt"
            model_path = Path(temp_dir) / "model.bin"
            input_path.touch()
            model_path.touch()
            windows = [
                VisionWindow(start=10.0, end=22.0, score=0.62, motion=7.2, saturation=32.0, scene_density=0.10),
                VisionWindow(start=46.0, end=58.0, score=0.85, motion=10.5, saturation=41.0, scene_density=0.24),
                VisionWindow(start=88.0, end=100.0, score=0.78, motion=9.2, saturation=37.0, scene_density=0.21),
            ]
            whisper_cues = [
                TranscriptionCue(
                    start=50.0,
                    end=53.0,
                    score=0.92,
                    text="Shutdown into Baron Nashor!",
                    keywords=("shutdown", "baron nashor"),
                )
            ]
            ocr_cues = [
                TranscriptionCue(
                    start=94.0,
                    end=96.0,
                    score=0.76,
                    text="Enemy ace secured",
                    keywords=("ace",),
                )
            ]
            with (
                patch("league_video_editor.cli._probe_duration_seconds", return_value=140.0),
                patch(
                    "league_video_editor.cli.detect_scene_events_adaptive",
                    return_value=([18.0, 52.0, 94.0], 0.20),
                ),
                patch("league_video_editor.cli.score_vision_activity", return_value=windows),
                patch("league_video_editor.cli._collect_local_ai_cues", return_value=whisper_cues),
                patch("league_video_editor.cli._collect_ocr_cues", return_value=ocr_cues),
            ):
                payload = generate_youtube_description(
                    input_path=input_path,
                    output_path=output_path,
                    champion="Zed",
                    channel_name="TestChannel",
                    scene_threshold=0.20,
                    vision_sample_fps=1.0,
                    vision_window_seconds=12.0,
                    vision_step_seconds=6.0,
                    whisper_model=model_path,
                    whisper_bin="whisper-cli",
                    whisper_language="en",
                    whisper_threads=4,
                    whisper_audio_stream=-1,
                    whisper_vad=True,
                    whisper_vad_threshold=0.50,
                    whisper_vad_model=None,
                    ocr_cue_scoring="auto",
                    tesseract_bin="tesseract",
                    ocr_sample_fps=0.25,
                    ocr_cue_threshold=0.16,
                    ai_cue_threshold=0.30,
                    max_moments=6,
                    title_count=3,
                )

            self.assertTrue(output_path.exists())
            output_text = output_path.read_text(encoding="utf-8")
            self.assertIn("Title Ideas:", output_text)
            self.assertIn("Description:", output_text)
            self.assertIn("Chapters:", output_text)
            self.assertIn("#LeagueOfLegends", output_text)
            self.assertGreaterEqual(len(payload["titles"]), 1)
            self.assertGreaterEqual(len(payload["chapters"]), 1)
            self.assertEqual(payload["whisper_cue_count"], 1)
            self.assertEqual(payload["ocr_cue_count"], 1)

    def test_render_highlights_retries_without_loudnorm_on_non_finite_audio(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            input_path.touch()
            output_path = Path(temp_dir) / "rendered.mp4"

            first_error = subprocess.CalledProcessError(
                returncode=1,
                cmd=["ffmpeg"],
                stderr="Input contains (near) NaN/+-Inf",
            )
            success = subprocess.CompletedProcess(args=["ffmpeg"], returncode=0)

            with (
                patch("league_video_editor.cli.read_plan", return_value=[Segment(0.0, 1.0)]),
                patch("league_video_editor.cli._has_audio_stream", return_value=True),
                patch(
                    "league_video_editor.cli._run_command_with_stderr_tail",
                    side_effect=[first_error, success],
                ) as run_with_stderr_tail,
                patch(
                    "league_video_editor.cli._run_command",
                    return_value=success,
                ) as run_command,
            ):
                render_highlights(
                    input_path=input_path,
                    plan_path=Path(temp_dir) / "plan.json",
                    output_path=output_path,
                    crf=20,
                    preset="medium",
                    video_encoder="libx264",
                    allow_upscale=False,
                    crossfade_seconds=0.0,
                    audio_fade_seconds=0.03,
                    two_pass_loudnorm=False,
                )

            self.assertEqual(run_with_stderr_tail.call_count, 2)
            self.assertEqual(run_command.call_count, 0)
            first_command = run_with_stderr_tail.call_args_list[0].args[0]
            second_command = run_with_stderr_tail.call_args_list[1].args[0]
            self.assertIn("loudnorm=I=-14:LRA=11:TP=-1.5", " ".join(first_command))
            self.assertNotIn("loudnorm=I=-14:LRA=11:TP=-1.5", " ".join(second_command))
            self.assertIn("-c:a", second_command)

    def test_transcode_full_match_retries_without_loudnorm_on_non_finite_audio(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            input_path.touch()
            output_path = Path(temp_dir) / "full.mp4"

            first_error = subprocess.CalledProcessError(
                returncode=1,
                cmd=["ffmpeg"],
                stderr="Input contains (near) NaN/+-Inf",
            )
            success = subprocess.CompletedProcess(args=["ffmpeg"], returncode=0)

            with (
                patch("league_video_editor.cli._has_audio_stream", return_value=True),
                patch(
                    "league_video_editor.cli._run_command_with_stderr_tail",
                    side_effect=[first_error, success],
                ) as run_with_stderr_tail,
                patch(
                    "league_video_editor.cli._run_command",
                    return_value=success,
                ) as run_command,
            ):
                transcode_full_match(
                    input_path=input_path,
                    output_path=output_path,
                    crf=20,
                    preset="medium",
                    video_encoder="libx264",
                    allow_upscale=False,
                    two_pass_loudnorm=False,
                )

            self.assertEqual(run_with_stderr_tail.call_count, 2)
            self.assertEqual(run_command.call_count, 1)
            first_command = run_with_stderr_tail.call_args_list[0].args[0]
            second_command = run_with_stderr_tail.call_args_list[1].args[0]
            self.assertIn("-af", first_command)
            self.assertIn("loudnorm=I=-14:LRA=11:TP=-1.5", " ".join(first_command))
            self.assertNotIn("loudnorm=I=-14:LRA=11:TP=-1.5", " ".join(second_command))
            self.assertIn("-c:a", second_command)

    def test_render_highlights_uses_two_pass_loudnorm_when_enabled(self) -> None:
        with TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            input_path.touch()
            output_path = Path(temp_dir) / "rendered.mp4"
            analysis_output = """
            {
              "input_i": "-23.1",
              "input_tp": "-1.2",
              "input_lra": "5.0",
              "input_thresh": "-34.0",
              "target_offset": "0.5"
            }
            """
            analysis_result = subprocess.CompletedProcess(
                args=["ffmpeg"], returncode=0, stdout="", stderr=analysis_output
            )
            encode_result = subprocess.CompletedProcess(
                args=["ffmpeg"], returncode=0, stdout="", stderr=""
            )

            with (
                patch("league_video_editor.cli.read_plan", return_value=[Segment(0.0, 1.0)]),
                patch("league_video_editor.cli._has_audio_stream", return_value=True),
                patch(
                    "league_video_editor.cli._run_command",
                    side_effect=[analysis_result],
                ) as run_command,
                patch(
                    "league_video_editor.cli._run_command_with_stderr_tail",
                    return_value=encode_result,
                ) as run_with_stderr_tail,
            ):
                render_highlights(
                    input_path=input_path,
                    plan_path=Path(temp_dir) / "plan.json",
                    output_path=output_path,
                    crf=20,
                    preset="medium",
                    video_encoder="libx264",
                    allow_upscale=False,
                    crossfade_seconds=0.0,
                    audio_fade_seconds=0.03,
                    two_pass_loudnorm=True,
                )

            self.assertEqual(run_command.call_count, 1)
            self.assertEqual(run_with_stderr_tail.call_count, 1)
            analysis_command = run_command.call_args_list[0].args[0]
            encode_command = run_with_stderr_tail.call_args_list[0].args[0]
            self.assertIn("print_format=json", " ".join(analysis_command))
            self.assertIn("measured_I=", " ".join(encode_command))

    def test_build_filter_complex_supports_crossfade_and_audio_fades(self) -> None:
        segments = [Segment(0.0, 5.0), Segment(8.0, 12.0)]
        filter_graph, map_video, map_audio = _build_filter_complex(
            segments,
            include_audio=True,
            include_video=True,
            allow_upscale=False,
            crossfade_seconds=0.2,
            audio_fade_seconds=0.05,
        )
        self.assertEqual(map_video, "[vout]")
        self.assertEqual(map_audio, "[aout]")
        self.assertIn("xfade=transition=fade", filter_graph)
        self.assertIn("acrossfade=d=0.200", filter_graph)
        self.assertIn("afade=t=in:st=0:d=0.050", filter_graph)

    def test_video_postprocess_filter_disables_upscale_by_default(self) -> None:
        filter_text = _video_postprocess_filter(allow_upscale=False)
        self.assertIn("if(gt(iw,1920),1920,iw)", filter_text)
        self.assertIn("if(gt(ih,1080),1080,ih)", filter_text)

    def test_video_codec_args_switch_for_hardware_encoder(self) -> None:
        args = _video_codec_args(video_encoder="h264_videotoolbox", crf=20, preset="medium")
        self.assertIn("h264_videotoolbox", args)
        self.assertIn("-b:v", args)
        self.assertNotIn("-crf", args)

    def test_extract_loudnorm_json_parses_required_fields(self) -> None:
        data = _extract_loudnorm_json(
            """
            random logs
            {
              "input_i": "-20.0",
              "input_tp": "-2.1",
              "input_lra": "6.5",
              "input_thresh": "-31.0",
              "target_offset": "0.1"
            }
            """
        )
        self.assertAlmostEqual(data["measured_I"], -20.0)
        self.assertAlmostEqual(data["measured_TP"], -2.1)
        self.assertAlmostEqual(data["measured_LRA"], 6.5)
        self.assertAlmostEqual(data["measured_thresh"], -31.0)
        self.assertAlmostEqual(data["offset"], 0.1)

    def test_extract_loudnorm_json_uses_last_matching_object(self) -> None:
        data = _extract_loudnorm_json(
            """
            {"debug":"value"}
            {
              "input_i": "-18.0",
              "input_tp": "-1.8",
              "input_lra": "4.5",
              "input_thresh": "-28.0",
              "target_offset": "0.2"
            }
            """
        )
        self.assertAlmostEqual(data["measured_I"], -18.0)
        self.assertAlmostEqual(data["offset"], 0.2)

    def test_loudnorm_error_matcher_accepts_variant_message(self) -> None:
        error = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg"],
            stderr="... Input contains NaN and +Inf samples ...",
        )
        self.assertTrue(_looks_like_loudnorm_nonfinite_error(error))

    def test_loudnorm_error_matcher_rejects_unrelated_error(self) -> None:
        error = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg"],
            stderr="Error while decoding stream #0:0",
        )
        self.assertFalse(_looks_like_loudnorm_nonfinite_error(error))

    def test_detect_scene_events_propagates_command_failures(self) -> None:
        with patch(
            "league_video_editor.cli._run_command",
            side_effect=subprocess.CalledProcessError(1, ["ffmpeg"]),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                detect_scene_events(Path("input.mp4"), scene_threshold=0.35)

    def test_detect_scene_events_adaptive_falls_back_to_lower_threshold(self) -> None:
        with patch(
            "league_video_editor.cli.detect_scene_events",
            side_effect=[[], [2.0, 8.0, 10.5]],
        ) as detect_scene_events_mock:
            events, threshold = detect_scene_events_adaptive(
                Path("input.mp4"),
                scene_threshold=0.35,
                duration_seconds=120.0,
                show_progress=False,
            )
        self.assertEqual(events, [2.0, 8.0, 10.5])
        self.assertLess(threshold, 0.35)
        self.assertEqual(detect_scene_events_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
