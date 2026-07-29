import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import json

import viewer


class TestViewerUploadWorkflow(unittest.TestCase):
    def test_upload_request_queues_full_quality_workflow(self):
        captured = {}

        class _Thread:
            def __init__(self, target, args, daemon):
                captured.update(target=target, args=args, daemon=daemon)

            def start(self):
                captured["started"] = True

        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            with patch.object(viewer.threading, "Thread", _Thread):
                with viewer.app.test_client() as client:
                    response = client.post("/analyze", json={
                        "video_path": video.name,
                        "skip_frames": 0,
                        "refresh_net_floor": True,
                    })

        self.assertEqual(response.status_code, 200)
        self.assertTrue(captured["started"])
        self.assertTrue(captured["daemon"])
        self.assertIs(captured["target"], viewer._run_analysis)
        self.assertEqual(captured["args"][2:], (0, True))

    def test_upload_defaults_to_refreshing_sam3_post_bases(self):
        captured = {}

        class _Thread:
            def __init__(self, target, args, daemon):
                captured["args"] = args

            def start(self):
                pass

        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            with patch.object(viewer.threading, "Thread", _Thread):
                with viewer.app.test_client() as client:
                    response = client.post("/analyze", json={"video_path": video.name})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["args"][2:], (0, True))

    def test_completed_job_is_restored_from_disk_after_restart(self):
        with tempfile.TemporaryDirectory() as root:
            output = Path(root) / "job_42"
            output.mkdir()
            (output / "clip_stats.json").write_text("{}", encoding="utf-8")
            (output / "clip_annotations.json").write_text(
                json.dumps({"original_video_path": "C:/games/clip.mp4"}), encoding="utf-8",
            )
            (output / "clip_analyzed.mp4").write_bytes(b"video")

            with patch.object(viewer, "RESULTS_DIR", Path(root)):
                viewer.jobs.pop("job_42", None)
                restored = viewer._get_job("job_42")

        self.assertEqual(restored["status"], "done")
        self.assertEqual(restored["video_stem"], "clip")
        self.assertEqual(restored["original_video_path"], "C:/games/clip.mp4")


if __name__ == "__main__":
    unittest.main()
