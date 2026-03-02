import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from league_video_editor.cli import (
    EditorError,
    Segment,
    build_segments,
    merge_segments,
    read_plan,
    sample_evenly,
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


if __name__ == "__main__":
    unittest.main()
