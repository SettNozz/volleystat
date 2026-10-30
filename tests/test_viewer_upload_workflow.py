import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import json

import viewer


class TestViewerUploadWorkflow(unittest.TestCase):
    def test_home_page_describes_cached_repeats_and_explicit_video_export(self):
        with viewer.app.test_client() as client:
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("first analysis saves model outputs", page)
        self.assertIn("Compatible repeats reuse", page)
        self.assertIn("encoded only when explicitly requested", page)
        self.assertIn('id="analysisProgressFill"', page)
        self.assertIn("job.frame_current", page)
        self.assertIn("job.processing_fps", page)
        self.assertIn("job.eta_sec", page)

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
        self.assertIn("renderAttackVectorMap", template)
        self.assertIn("ATTACKING SIDE (mirrored above the net)", template)
        self.assertIn("2 * topY - floorY1", template)
        self.assertIn("toggleCourtEdit", template)
        self.assertIn("saveCourtCorrection", template)
        # Corners routinely fall outside a wide-lens frame, so the lines have
        # to be draggable on their own.
        self.assertIn("cornersFromEdgeAnchors", template)
        self.assertIn("clipSegmentToFrame", template)
        self.assertIn("off-frame", template)
        self.assertNotIn("createRadialGradient", template)
        self.assertNotIn('type="video/mp4"', template)
        self.assertIn("Object.prototype.hasOwnProperty.call(pose, 'player_id')", template)
        self.assertIn("pose.player_id || 'P?'", template)
        self.assertIn("Number(ann.f) !== currentFrameIdx", template)
        self.assertIn("player IDs unavailable", template)

    def test_lifecycle_stats_keep_only_confirmed_playable_rallies(self):
        stats = viewer._prepare_viewer_stats({
            "total_rallies": 99,
            "rally_lifecycle": {
                "service_attempts": 8,
                "confirmed_serves": 7,
                "playable_rallies": 4,
                "aces_no_reception": 1,
                "reception_failed": 2,
                "serve_faults": 1,
            },
            "rallies": [
                {
                    "start_frame": 100,
                    "lifecycle": {
                        "status": "playable_rally",
                        "is_playable": True,
                    },
                },
                {
                    "start_frame": 200,
                    "lifecycle": {
                        "status": "ace_no_reception",
                        "is_playable": False,
                    },
                },
                {"start_frame": 300},
            ],
            "service_attempts": [{"lifecycle": {"status": "serve_fault"}}],
        })

        self.assertTrue(stats["viewer_lifecycle"]["available"])
        self.assertEqual(stats["total_rallies"], 4)
        self.assertEqual(stats["viewer_lifecycle"]["service_attempts"], 8)
        self.assertEqual([r["start_frame"] for r in stats["rallies"]], [100])
        self.assertEqual(len(stats["service_attempts"]), 1)

    def test_legacy_stats_preserve_rallies_without_guessing_lifecycle(self):
        legacy_rallies = [{"start_frame": 10}, {"start_frame": 50}]
        stats = viewer._prepare_viewer_stats({
            "total_rallies": 2,
            "rallies": legacy_rallies,
        })

        self.assertFalse(stats["viewer_lifecycle"]["available"])
        self.assertEqual(stats["viewer_lifecycle"]["playable_rallies"], 2)
        self.assertIsNone(stats["viewer_lifecycle"]["service_attempts"])
        self.assertEqual(stats["rallies"], legacy_rallies)

    def test_results_template_consumes_canonical_lifecycle_metadata(self):
        template = (
            Path(viewer.app.template_folder) / "viewer.html"
        ).read_text(encoding="utf-8")

        self.assertIn("Service attempts", template)
        self.assertIn("Confirmed serves", template)
        self.assertIn("Aces / no reception", template)
        self.assertIn("Reception failed", template)
        self.assertIn("Serve faults", template)
        self.assertIn("const SERVICE_ATTEMPTS", template)
        self.assertIn("RAW_RALLIES.filter(isPlayableRally)", template)
        self.assertIn("lc.status === 'playable_rally'", template)
        self.assertIn("Confirmed playable rally", template)
        self.assertIn("rallyStartFrame(r)", template)
        self.assertIn("const lc = ann.lifecycle", template)
        self.assertIn("RECEPTION CONFIRMED", template)

    def test_results_page_renders_lifecycle_counts_and_attempts(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            stats_path = root / "clip_stats.json"
            annotations_path = root / "clip_annotations.json"
            video_path = root / "clip.mp4"
            stats_path.write_text(json.dumps({
                "fps": 30,
                "total_frames": 300,
                "total_rallies": 9,
                "ball_detection_rate": 0.8,
                "processing_time_sec": 12,
                "all_events": [],
                "rallies": [{
                    "start_frame": 30,
                    "duration_sec": 3,
                    "contacts_team_a": 2,
                    "contacts_team_b": 2,
                    "events": [],
                    "lifecycle": {
                        "status": "playable_rally",
                        "is_playable": True,
                        "start_frame": 30,
                    },
                }],
                "service_attempts": [{
                    "lifecycle": {
                        "status": "serve_fault",
                        "is_playable": False,
                        "serve_confirmed": False,
                        "serve_frame": 200,
                        "serving_team": "team_b",
                    },
                }],
                "rally_lifecycle": {
                    "service_attempts": 6,
                    "confirmed_serves": 5,
                    "playable_rallies": 3,
                    "aces_no_reception": 1,
                    "reception_failed": 1,
                    "serve_faults": 1,
                },
            }), encoding="utf-8")
            annotations_path.write_text("{}", encoding="utf-8")
            video_path.write_bytes(b"video")
            viewer.jobs["lifecycle_test"] = {
                "status": "done",
                "stats_path": str(stats_path),
                "annotations_path": str(annotations_path),
                "video_path": str(video_path),
                "video_stem": "clip",
                "job_id": "lifecycle_test",
                "original_video_path_url": "",
            }
            try:
                with viewer.app.test_client() as client:
                    response = client.get("/results/lifecycle_test")
            finally:
                viewer.jobs.pop("lifecycle_test", None)

        self.assertEqual(response.status_code, 200)
        page = response.get_data(as_text=True)
        self.assertIn("3 playable rallies", page)
        self.assertIn("Service attempts", page)
        self.assertIn("Confirmed serves", page)
        self.assertIn("serve_fault", page)

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

    def test_viewer_saves_per_result_court_correction(self):
        with tempfile.TemporaryDirectory() as root:
            annotations_path = Path(root) / "clip_annotations.json"
            annotations_path.write_text(json.dumps({
                "frame_width": 1000,
                "frame_height": 600,
                "court_corners": [[1, 2], [3, 4], [5, 6], [7, 8]],
            }), encoding="utf-8")
            viewer.jobs["court_test"] = {
                "status": "done",
                "annotations_path": str(annotations_path),
            }
            try:
                with viewer.app.test_client() as client:
                    response = client.put("/api/court/court_test", json={
                        "court_corners": [[100, 200], [900, 200], [980, 580], [20, 580]],
                        "net_y": 240,
                        "net_x_left": 100,
                        "net_x_right": 900,
                    })
            finally:
                viewer.jobs.pop("court_test", None)

            saved = json.loads(annotations_path.read_text(encoding="utf-8"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(saved["court_corners"][0], [100.0, 200.0])
        self.assertEqual(saved["net_y"], 240.0)
        self.assertEqual(saved["manual_court_correction"]["source"], "viewer")

    def _court_job(self, root, annotations):
        annotations_path = Path(root) / "clip_annotations.json"
        annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
        stats_path = Path(root) / "clip_stats.json"
        stats_path.write_text(json.dumps({"all_events": []}), encoding="utf-8")
        viewer.jobs["court_test"] = {
            "status": "done",
            "annotations_path": str(annotations_path),
            "stats_path": str(stats_path),
        }
        self.addCleanup(viewer.jobs.pop, "court_test", None)
        return annotations_path

    def test_court_corner_outside_the_frame_is_accepted(self):
        # A wide lens puts the near-left corner off the picture; the user can
        # only place it by dragging the two lines that meet there.
        with tempfile.TemporaryDirectory() as root:
            annotations_path = self._court_job(root, {
                "frame_width": 1000, "frame_height": 600, "fps": 30, "frames": [],
            })
            with patch.object(viewer, "COURT_SETUPS_PATH", Path(root) / "setups.json"):
                with viewer.app.test_client() as client:
                    response = client.put("/api/court/court_test", json={
                        "court_corners": [[300, 200], [700, 200], [1400, 580], [-360, 575]],
                    })
            saved = json.loads(annotations_path.read_text(encoding="utf-8"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(saved["court_corners"][3], [-360.0, 575.0])
        self.assertEqual(saved["attack_heatmap"]["version"], 4)

    def test_self_intersecting_court_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            self._court_job(root, {
                "frame_width": 1000, "frame_height": 600, "fps": 30, "frames": [],
            })
            with viewer.app.test_client() as client:
                response = client.put("/api/court/court_test", json={
                    "court_corners": [[100, 100], [900, 100], [100, 500], [900, 500]],
                })

        self.assertEqual(response.status_code, 400)
        self.assertIn("convex", response.get_json()["error"])

    def test_recompute_upgrades_an_older_attack_map_in_place(self):
        with tempfile.TemporaryDirectory() as root:
            annotations_path = self._court_job(root, {
                "frame_width": 1000, "frame_height": 600, "fps": 30, "frames": [],
                "court_corners": [[300, 200], [700, 200], [1400, 580], [-360, 575]],
                "attack_heatmap": {"version": 2, "hits": [], "zones": []},
            })
            with viewer.app.test_client() as client:
                state = client.get("/api/court/court_test").get_json()
                response = client.post("/api/court/court_test/recompute")
            saved = json.loads(annotations_path.read_text(encoding="utf-8"))

        self.assertEqual(state["heatmap_version"], 2)
        self.assertEqual(state["current_heatmap_version"], 4)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(saved["attack_heatmap"]["version"], 4)

    def test_court_setups_are_saved_listed_and_deleted(self):
        corners = [[300, 200], [700, 200], [1400, 580], [-360, 575]]
        with tempfile.TemporaryDirectory() as root:
            with patch.object(viewer, "COURT_SETUPS_PATH", Path(root) / "setups.json"):
                with viewer.app.test_client() as client:
                    created = client.post("/api/court_setups", json={
                        "name": "Gym A", "court_corners": corners,
                        "frame_width": 1000, "frame_height": 600,
                    }).get_json()
                    listed = client.get("/api/court_setups").get_json()
                    deleted = client.delete("/api/court_setups/" + created["id"])
                    missing = client.delete("/api/court_setups/" + created["id"])

        self.assertEqual([setup["name"] for setup in listed["setups"]], ["Gym A"])
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(missing.status_code, 404)


class TestViewerResultsLibrary(unittest.TestCase):
    """The library lists analysed results so a run can be reopened by name."""

    def _fixture_root(self, root: Path) -> None:
        results = root / "results"
        source = root / "data" / "game.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"video")

        full = results / "full_run"
        full.mkdir(parents=True)
        (full / "game_stats.json").write_text(json.dumps({
            "video": "game.mp4",
            "fps": 30.0,
            "total_frames": 900,
            "ball_detection_rate": 0.51,
            "total_rallies": 9,
            "all_events": [{}, {}, {}],
            "rally_lifecycle": {"playable_rallies": 4},
            "match_analytics": {"sets": []},
            "processing_time_sec": 120.0,
        }), encoding="utf-8")
        (full / "game_annotations.json").write_text(
            json.dumps({"original_video_path": str(source), "frames": []}),
            encoding="utf-8",
        )
        (full / "game_analyzed.mp4").write_bytes(b"annotated")

        job = results / "viewer" / "job_7"
        job.mkdir(parents=True)
        (job / "clip_stats.json").write_text(json.dumps({
            "fps": 30.0, "total_frames": 300, "all_events": [],
        }), encoding="utf-8")
        (job / "clip_annotations.json").write_text(
            json.dumps({"original_video_path": str(source)}), encoding="utf-8",
        )

        # Reconstruction reviews export annotations only; the viewer cannot open
        # them, so the library has to list them as blocked instead of hiding them.
        partial = results / "reconstruction_only"
        partial.mkdir(parents=True)
        (partial / "game_annotations.json").write_text("{}", encoding="utf-8")

    def _index(self, root: Path) -> dict:
        with patch.object(viewer, "PROJECT_ROOT", root), \
                patch.object(viewer, "RESULTS_ROOT", root / "results"), \
                patch.object(viewer, "RESULTS_DIR", root / "results" / "viewer"):
            return {entry["results_dir"]: entry for entry in viewer._library_index()}

    def test_index_summarises_every_result_and_flags_missing_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture_root(root)
            index = self._index(root)

        self.assertEqual(
            sorted(index),
            ["results/full_run", "results/reconstruction_only", "results/viewer/job_7"],
        )

        full = index["results/full_run"]
        self.assertEqual(full["stem"], "game")
        self.assertTrue(full["has_annotated_video"])
        self.assertTrue(full["playable"])
        self.assertEqual(
            full["open_url"],
            "/load?video=game&results_dir=results%2Ffull_run",
        )
        # The canonical lifecycle counter wins over the legacy rally total.
        self.assertEqual(full["summary"]["total_rallies"], 4)
        self.assertEqual(full["summary"]["total_events"], 3)
        self.assertEqual(full["summary"]["duration_sec"], 30.0)
        self.assertTrue(full["summary"]["has_match_analytics"])
        self.assertFalse(full["summary"]["has_reconstruction"])

        # A persisted upload restores through its job route, not by path.
        self.assertEqual(index["results/viewer/job_7"]["open_url"], "/results/job_7")

        blocked = index["results/reconstruction_only"]
        self.assertIsNone(blocked["open_url"])
        self.assertIn("game_stats.json", blocked["blocked_reason"])

    def test_index_reads_source_path_without_parsing_whole_annotations(self):
        # Annotation files run to tens of megabytes, so the library reads only
        # the header. A truncated file still has to yield its source video.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture_root(root)
            source = root / "data" / "game.mp4"
            annotations = root / "results" / "full_run" / "game_annotations.json"
            annotations.write_text(
                '{"fps":30.0,"original_video_path":'
                + json.dumps(str(source))
                + ',"frames":[{"f":0,"ball":[1.0,2.0]},{"f":1,',
                encoding="utf-8",
            )
            index = self._index(root)

        entry = index["results/full_run"]
        self.assertEqual(entry["original_video"], str(source))
        self.assertTrue(entry["playable"])

    def test_library_api_and_page_expose_the_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture_root(root)
            with patch.object(viewer, "PROJECT_ROOT", root), \
                    patch.object(viewer, "RESULTS_ROOT", root / "results"), \
                    patch.object(viewer, "RESULTS_DIR", root / "results" / "viewer"):
                with viewer.app.test_client() as client:
                    api = client.get("/api/library")
                    page = client.get("/library")

        payload = api.get_json()
        self.assertEqual(api.status_code, 200)
        self.assertEqual(payload["root"], "results")
        self.assertEqual(payload["count"], 3)
        self.assertEqual(len(payload["entries"]), 3)
        # Newest first keeps the run a user just finished at the top.
        modified = [entry["modified"] for entry in payload["entries"]]
        self.assertEqual(modified, sorted(modified, reverse=True))

        self.assertEqual(page.status_code, 200)
        markup = page.get_data(as_text=True)
        self.assertIn("/api/library", markup)
        self.assertIn('id="onlyOpenable"', markup)
        self.assertIn("Open in viewer", markup)

    def test_load_route_resolves_relative_results_dir_from_project_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._fixture_root(root)
            with patch.object(viewer, "PROJECT_ROOT", root):
                with viewer.app.test_client() as client:
                    response = client.get(
                        "/load", query_string={
                            "video": "game", "results_dir": "results/full_run",
                        },
                    )

        self.assertEqual(response.status_code, 302)
        self.assertIn("/results/preloaded_game", response.headers["Location"])
        viewer.jobs.pop("preloaded_game", None)

    def test_upload_analysis_runs_the_contact_spotter_when_installed(self):
        captured = []

        class _Analyzer:
            def __init__(self, **kwargs):
                captured.append(kwargs)

            def process_video(self, video_path, output_dir):
                return None

        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "spotter.pt"
            checkpoint.write_bytes(b"checkpoint")
            with patch("src.pipeline.game_analyzer.GameAnalyzer", _Analyzer), \
                    patch.object(viewer, "RESULTS_DIR", Path(root)):
                with patch.object(viewer, "CONTACT_SPOTTER_PATH", checkpoint):
                    viewer._run_analysis("spotter_on", str(Path(root) / "clip.mp4"), 0)
                with patch.object(viewer, "CONTACT_SPOTTER_PATH", Path(root) / "missing.pt"):
                    viewer._run_analysis("spotter_off", str(Path(root) / "clip.mp4"), 0)

        self.assertEqual(captured[0]["contact_spotter_path"], str(checkpoint))
        # Without the checkpoint the upload still runs, on the FSM's events.
        self.assertIsNone(captured[1]["contact_spotter_path"])
        self.assertEqual(viewer.jobs.pop("spotter_off")["status"], "done")
        viewer.jobs.pop("spotter_on", None)

    def test_frame_thumbnail_returns_the_requested_frame(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "clip.avi"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 360))
            for value in range(0, 250, 25):
                writer.write(np.full((360, 640, 3), value, dtype=np.uint8))
            writer.release()
            viewer.jobs["thumb_test"] = {
                "status": "done", "job_id": "thumb_test",
                "video_path": str(Path(root) / "missing_analyzed.mp4"),
                "original_video_path": str(source),
            }
            try:
                with viewer.app.test_client() as client:
                    response = client.get("/api/frame/thumb_test/4")
                    beyond = client.get("/api/frame/thumb_test/500")
                    unknown = client.get("/api/frame/no_such_job/4")
            finally:
                viewer.jobs.pop("thumb_test", None)
                viewer._thumb_cache.clear()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/jpeg")
        image = cv2.imdecode(np.frombuffer(response.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(image.shape, (180, 320))
        self.assertAlmostEqual(float(image.mean()), 100.0, delta=6.0)
        self.assertEqual(beyond.status_code, 404)
        self.assertEqual(unknown.status_code, 404)

    def test_rally_clips_show_their_first_frame(self):
        template = (Path(viewer.__file__).parent / "viewer" / "templates" / "viewer.html").read_text(encoding="utf-8")
        self.assertIn("/api/frame/${JOB_ID}/${rallyStartFrame(r)}", template)


if __name__ == "__main__":
    unittest.main()
