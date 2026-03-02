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
    _build_filter_complex,
    _extract_loudnorm_json,
    _looks_like_loudnorm_nonfinite_error,
    _probe_duration_seconds,
    _validate_cli_options,
    _video_codec_args,
    _video_postprocess_filter,
    build_segments,
    detect_scene_events,
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
        self.assertEqual(len(segments), 3)
        self.assertGreater(segments[0].duration, 0.0)

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

    def test_validate_cli_options_rejects_non_finite_scene_threshold(self) -> None:
        args = argparse.Namespace(
            command="analyze",
            scene_threshold=float("nan"),
            clip_before=8.0,
            clip_after=12.0,
            min_gap_seconds=18.0,
            max_clips=20,
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
            crf=20,
            crossfade_seconds=0.0,
            audio_fade_seconds=0.03,
            video_encoder="libx264",
        )
        with self.assertRaisesRegex(EditorError, "scene-threshold must be between 0.0 and 1.0"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_crf_out_of_range(self) -> None:
        args = argparse.Namespace(command="full", crf=52, video_encoder="libx264")
        with self.assertRaisesRegex(EditorError, "crf must be between 0 and 51"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_negative_crossfade(self) -> None:
        args = argparse.Namespace(
            command="render",
            crf=20,
            crossfade_seconds=-0.1,
            audio_fade_seconds=0.03,
            video_encoder="libx264",
        )
        with self.assertRaisesRegex(EditorError, "crossfade-seconds"):
            _validate_cli_options(args)

    def test_validate_cli_options_rejects_invalid_video_encoder(self) -> None:
        args = argparse.Namespace(
            command="full",
            crf=20,
            video_encoder="not-real",
        )
        with self.assertRaisesRegex(EditorError, "video-encoder must be one of"):
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
                patch("league_video_editor.cli._run_command", side_effect=[first_error, success]) as run_command,
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

            self.assertEqual(run_command.call_count, 2)
            first_command = run_command.call_args_list[0].args[0]
            first_kwargs = run_command.call_args_list[0].kwargs
            second_command = run_command.call_args_list[1].args[0]
            self.assertTrue(first_kwargs.get("capture_output", False))
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
                patch("league_video_editor.cli._run_command", side_effect=[first_error, success]) as run_command,
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

            self.assertEqual(run_command.call_count, 2)
            first_command = run_command.call_args_list[0].args[0]
            first_kwargs = run_command.call_args_list[0].kwargs
            second_command = run_command.call_args_list[1].args[0]
            self.assertTrue(first_kwargs.get("capture_output", False))
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
                    side_effect=[analysis_result, encode_result],
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
                    two_pass_loudnorm=True,
                )

            self.assertEqual(run_command.call_count, 2)
            analysis_command = run_command.call_args_list[0].args[0]
            encode_command = run_command.call_args_list[1].args[0]
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


if __name__ == "__main__":
    unittest.main()
