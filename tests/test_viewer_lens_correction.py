"""Lens preview, calibration binding and prepared-source analysis dispatch."""

import json
from unittest.mock import patch

import cv2
import numpy as np
import pytest

import viewer
from src.video.lens_correction import LensCorrection, rectify_frame


@pytest.fixture
def scene(tmp_path):
    source = tmp_path / "wide.mp4"
    source.write_bytes(b"source")
    frame = np.random.default_rng(2).integers(0, 256, (96, 160, 3), dtype=np.uint8)
    profile = LensCorrection(enabled=True).to_dict()
    edit = {"frame_width": 160, "frame_height": 96,
            "court_corners": [[30, 30], [130, 30], [155, 90], [5, 90]],
            "net_tape": [[20, 25], [140, 25]], "lens_correction": profile}
    return source, frame, profile, edit


def test_native_preview_uses_correction_and_does_not_reuse_raw_geometry(scene, tmp_path):
    source, frame, profile, _ = scene
    (tmp_path / "court_calibration.json").write_text(json.dumps({
        "frame_width": 160, "frame_height": 96,
        "court_corners": [[30, 30], [130, 30], [155, 90], [5, 90]],
    }))
    with patch.object(viewer, "PROJECT_ROOT", tmp_path), \
            patch.object(viewer, "_calibration_frame", return_value=(frame, 10)), viewer.app.test_client() as client:
        preview = client.post("/api/analysis_calibration", json={
            "video_path": str(source), "lens_correction": profile,
        })
        assert preview.status_code == 200
        assert preview.json["court_corners"] is None
        assert preview.json["lens_correction"] == profile
        native = client.get("/api/analysis_calibration/frame", query_string={
            "video_path": str(source), "lens_correction": json.dumps(profile),
        })
    assert native.status_code == 200
    _, expected = cv2.imencode(".jpg", rectify_frame(frame, LensCorrection.from_dict(profile)),
                               [cv2.IMWRITE_JPEG_QUALITY, 92])
    assert native.data == expected.tobytes()


@pytest.mark.parametrize("change", ["no_calibration", "stale_profile", "correction_disabled", "invalid_calibration"])
def test_analysis_refuses_unpreviewed_or_mismatched_correction(scene, change):
    source, frame, profile, edit = scene
    payload = {"video_path": str(source), "lens_correction": profile, "calibration": edit}
    if change == "no_calibration":
        payload.pop("calibration")
    elif change == "stale_profile":
        payload["lens_correction"] = {**profile, "balance": .9}
    elif change == "correction_disabled":
        payload.pop("lens_correction")
    else:
        payload["calibration"] = "invalid"
    with patch.object(viewer, "_calibration_frame", return_value=(frame, 10)), \
            patch.object(viewer.threading, "Thread") as thread, viewer.app.test_client() as client:
        response = client.post("/analyze", json=payload)
    assert response.status_code == 400
    thread.assert_not_called()


def test_corrected_source_is_shared_by_analysis_and_playback(scene, tmp_path):
    source, frame, profile, edit = scene
    prepared = tmp_path / "prepared" / "wide.mp4"
    prepared.parent.mkdir()
    prepared.write_bytes(b"corrected")
    provenance = {"source_video_path": str(source), "output_video_path": str(prepared)}
    with patch.object(viewer, "_calibration_frame", return_value=(frame, 10)), \
            patch.object(viewer.threading, "Thread") as thread, viewer.app.test_client() as client:
        response = client.post("/analyze", json={
            "video_path": str(source), "lens_correction": profile, "calibration": edit,
        })
    assert response.status_code == 200
    queued = thread.call_args.kwargs["args"]
    assert queued[-1] == profile

    def prepare(*args, **kwargs):
        kwargs["progress"]({"frame_current": 90, "frame_total": 100, "progress": 90})
        return prepared, provenance

    instances = []

    def build_analyzer(**kwargs):
        from unittest.mock import Mock
        assert "frame_current" not in viewer.jobs[queued[0]]
        assert viewer.jobs[queued[0]]["progress"] == 0
        instance = Mock()
        instances.append(instance)
        return instance

    with patch.object(viewer, "RESULTS_DIR", tmp_path), \
            patch.object(viewer, "prepare_video", side_effect=prepare), \
            patch("src.pipeline.game_analyzer.GameAnalyzer", side_effect=build_analyzer) as analyzer:
        try:
            viewer._run_analysis(*queued)
            assert instances[0].process_video.call_args.kwargs["video_path"] == str(prepared)
            assert viewer.jobs[queued[0]]["original_video_path"] == str(prepared)
            assert viewer.jobs[queued[0]]["status"] == "done"
            options = analyzer.call_args.kwargs
            calibration = json.loads(open(options["calibration_path"], encoding="utf-8").read())
            assert calibration["lens_correction"] == profile
            assert calibration["coordinate_space"] == "rectified_image_pixels"
            assert json.loads((tmp_path / queued[0] / "video_preprocessing.json").read_text()) == provenance
        finally:
            viewer.jobs.pop(queued[0], None)
