import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import json

import viewer


class TestViewerUploadWorkflow(unittest.TestCase):
    def test_home_page_shows_measured_4k_processing_time(self):
        with viewer.app.test_client() as client:
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("57s 4K video takes ~3 min 30s", page)
        self.assertIn("Every frame + batch 8", page)

    def test_results_template_guards_jumps_and_draws_three_attack_zones(self):
        template = (
            Path(viewer.app.template_folder) / "viewer.html"
        ).read_text(encoding="utf-8")

        self.assertIn("step > MAX_TRAIL_STEP", template)
        self.assertIn("const GAP_BREAK = 5", template)
        self.assertIn("p1.newTrack", template)
        self.assertIn("Attack origins", template)
        self.assertIn("const ids = ['left', 'center', 'right']", template)
        self.assertIn('id="togCourt"', template)
        self.assertIn("window._annData.court_corners", template)
        self.assertNotIn("createRadialGradient", template)

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
                        "ball_batch_size": 8,
                        "net_profile": "mixed",
                    })

        self.assertEqual(response.status_code, 200)
        self.assertTrue(captured["started"])
        self.assertTrue(captured["daemon"])
        self.assertIs(captured["target"], viewer._run_analysis)
        self.assertEqual(captured["args"][2:], (0, True, 8, 2.34))

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
        self.assertEqual(captured["args"][2:], (0, True, 8, 2.34))

    def test_upload_can_select_mens_net_height(self):
        captured = {}

        class _Thread:
            def __init__(self, target, args, daemon):
                captured["args"] = args

            def start(self):
                pass

        with tempfile.NamedTemporaryFile(suffix=".mp4") as video:
            with patch.object(viewer.threading, "Thread", _Thread):
                with viewer.app.test_client() as client:
                    response = client.post("/analyze", json={
                        "video_path": video.name,
                        "net_profile": "men",
                    })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["args"][-1], 2.43)

    def test_relative_video_path_is_resolved_from_project_root(self):
        captured = {}

        class _Thread:
            def __init__(self, target, args, daemon):
                captured["args"] = args

            def start(self):
                pass

        with tempfile.TemporaryDirectory() as root:
            video = Path(root) / "data" / "clip.mp4"
            video.parent.mkdir()
            video.write_bytes(b"video")
            with patch.object(viewer, "PROJECT_ROOT", Path(root)):
                with patch.object(viewer.threading, "Thread", _Thread):
                    with viewer.app.test_client() as client:
                        response = client.post(
                            "/analyze", json={"video_path": "data/clip.mp4"},
                        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(Path(captured["args"][1]), video)

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

    def test_completed_fast_job_restores_without_annotated_video(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "clip.mp4"
            source.write_bytes(b"video")
            output = Path(root) / "job_43"
            output.mkdir()
            (output / "clip_stats.json").write_text("{}", encoding="utf-8")
            (output / "clip_annotations.json").write_text(
                json.dumps({"original_video_path": str(source)}), encoding="utf-8",
            )

            with patch.object(viewer, "RESULTS_DIR", Path(root)):
                viewer.jobs.pop("job_43", None)
                restored = viewer._get_job("job_43")

        self.assertEqual(restored["status"], "done")
        self.assertEqual(Path(restored["video_path"]), source)


if __name__ == "__main__":
    unittest.main()
