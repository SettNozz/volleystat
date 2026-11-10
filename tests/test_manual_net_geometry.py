"""Manual net geometry must survive the preview -> pipeline -> render boundary."""

import json
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.pipeline.game_analyzer import GameAnalyzer, GameStats
from src.viewer_app.calibration_preview import _manual_calibration


EDIT = {
    "frame_width": 3840, "frame_height": 2160,
    "court_corners": [[1000, 950], [2800, 990], [4100, 2050], [-200, 1980]],
    "net_tape": [[1200.25, 820.5], [2800.75, 832.25]],
}


def _browser(payload):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required to exercise the shipped viewer scripts")
    runner = Path(__file__).parent / "js" / "manual_net_geometry.cjs"
    result = subprocess.run(
        [node, str(runner)], input=json.dumps(payload), text=True,
        capture_output=True, check=True, timeout=15,
    )
    return json.loads(result.stdout)


def _load(tmp_path, edit=EDIT, net_height=2.34):
    source = tmp_path / "court_calibration.json"
    source.write_text(json.dumps(_manual_calibration(edit, 3840, 2160, net_height)))
    analyzer = GameAnalyzer.__new__(GameAnalyzer)
    analyzer.calibration = analyzer._load_calibration(str(source))
    analyzer.net_y_override = None
    analyzer.net_height_m = net_height
    analyzer._projector = analyzer._projector_calib = None
    return analyzer


@pytest.mark.parametrize("size", [(3840, 2160), (1280, 720), (1600, 1000)])
@pytest.mark.parametrize("profile,height", [("mixed", 2.34), ("men", 2.43)])
def test_dragged_preview_matches_all_three_exported_lines(tmp_path, size, profile, height):
    preview = _browser({
        "mode": "preview", "data": {**EDIT, "setups": [], "image": "preview.jpg"},
        "profile": profile, "target": [2900.125, 840.875],
        "rect": {"left": 73.5, "top": 48.25, "width": 1000.5, "height": 562.78125},
    })
    edit = preview["edit"]
    np.testing.assert_allclose(edit["net_tape"], [[1200.25, 820.5], [2900.125, 840.875]])
    assert preview["after"] == edit  # submitted geometry is a snapshot

    analyzer = _load(tmp_path, edit, height)
    original = asdict(analyzer.calibration)
    source_bytes = (tmp_path / "court_calibration.json").read_bytes()
    scaled = analyzer._scale_calibration(*size)
    stats = GameStats(video_name="clip", player_contacts={"enabled": False})
    analyzer._save_and_print(stats, tmp_path, "clip", calibration_override=scaled)
    exported = json.loads((tmp_path / "clip_annotations.json").read_text())
    assert asdict(analyzer.calibration) == original
    assert (tmp_path / "court_calibration.json").read_bytes() == source_bytes

    sx, sy = size[0] / 3840, size[1] / 2160
    top = np.asarray(edit["net_tape"])
    floor = np.asarray([original["net_floor_left"], original["net_floor_right"]])
    bottom = top + [0, original["net_tape_bottom"] - original["net_tape_top"]]
    preview_colors = ["#ffcb55", "#ff8c00", "#ff00ff"]
    overlay_colors = ["rgba(255, 255, 255, 0.95)", "rgba(255, 140, 0, 0.9)", "rgba(255, 0, 255, 0.92)"]
    rendered = _browser({"mode": "overlay", "data": exported, "canvas": {"width": 960, "height": 540}})
    for expected, preview_color, overlay_color in zip((top, bottom, floor), preview_colors, overlay_colors):
        before = next(s["points"] for s in preview["strokes"] if s["color"] == preview_color)
        after = next(s["points"] for s in rendered["strokes"] if s["color"] == overlay_color)
        np.testing.assert_allclose(before, expected, atol=1e-7)
        np.testing.assert_allclose(after, expected * [960 / 3840, 540 / 2160], atol=1e-7)
    np.testing.assert_allclose([exported["net_x_left"], exported["net_x_right"]], top[:, 0] * sx)
    np.testing.assert_allclose(exported["net_tape_endpoint_y"], top[:, 1] * sy)


def test_manual_net_skips_detection_even_when_refresh_was_requested(tmp_path):
    analyzer = _load(tmp_path)
    analyzer.enable_net_detection = True
    before = asdict(analyzer.calibration)
    with patch.object(analyzer, "_run_net_detection_uncached") as detection:
        analyzer._maybe_run_sam_net_detection("clip.webm", 3840, 2160, refresh_floor=True)
    detection.assert_not_called()
    with patch("src.detection.net_detector.NetDetector") as detector:
        analyzer._run_net_detection_uncached("clip.webm", 3840, 2160, refresh_floor=True)
    detector.assert_not_called()
    assert asdict(analyzer.calibration) == before


def test_viewer_save_and_reload_keeps_all_three_corrected_lines(tmp_path):
    import viewer

    analyzer = _load(tmp_path)
    analyzer._save_and_print(GameStats(player_contacts={"enabled": False}), tmp_path, "clip")
    path = tmp_path / "clip_annotations.json"
    source_bytes = (tmp_path / "court_calibration.json").read_bytes()
    job = {"status": "done", "annotations_path": str(path),
           "stats_path": str(tmp_path / "clip_stats.json")}
    before = json.loads(path.read_text())
    change = {"court_corners": EDIT["court_corners"], "net_x_left": 1380.25,
              "net_x_right": 2980.75, "net_y": before["net_y"] + 5}
    with patch.dict(viewer.jobs, manual_net_test=job), \
            patch.object(viewer, "_setup_store") as store, \
            viewer.app.test_client() as client:
        store.return_value.load.return_value = []
        saved = client.put("/api/court/manual_net_test", json=change)
        assert saved.status_code == 200
        reloaded = client.get("/api/court/manual_net_test").get_json()
        saved_bytes = path.read_bytes()
        invalid = client.put("/api/court/manual_net_test", json={**change, "net_x_right": 0})
        assert invalid.status_code == 400
        assert path.read_bytes() == saved_bytes
    payload = json.loads(path.read_text())
    for x, key in ((1380.25, "net_floor_left"), (2980.75, "net_floor_right")):
        assert payload[key][0] == pytest.approx(x)
        assert reloaded[key] == saved.get_json()[key] == payload[key]
    np.testing.assert_allclose(payload["net_tape_endpoint_y"], np.asarray(before["net_tape_endpoint_y"]) + 5)
    assert reloaded["net_tape_endpoint_y"] == payload["net_tape_endpoint_y"]
    assert reloaded["net_tape_bottom"] == payload["net_tape_bottom"]
    assert payload["frames"] == before["frames"]
    assert (tmp_path / "court_calibration.json").read_bytes() == source_bytes


@pytest.mark.parametrize("manual", [True, False])
def test_opencv_and_viewer_use_tape_x_for_manual_and_posts_for_automatic(tmp_path, manual):
    analyzer = _load(tmp_path)
    c = analyzer.calibration
    # A floor line from another capture must not replace explicit tape clicks.
    analyzer.calibration = replace(
        c, net_floor_left=(c.net_x_left - 180, c.net_floor_left[1]),
        net_floor_right=(c.net_x_right - 180, c.net_floor_right[1]),
        net_tape_endpoint_y=c.net_tape_endpoint_y if manual else None,
    )
    c = analyzer.calibration
    left = c.net_x_left if manual else c.net_floor_left[0]
    right = c.net_x_right if manual else c.net_floor_right[0]
    ys = c.net_tape_endpoint_y or (c.net_tape_top, c.net_tape_top)
    strokes = []
    with patch("src.pipeline.analyzer.rendering.cv2.line", side_effect=lambda image, a, b, *args: strokes.append((a, b))):
        analyzer._draw_frame(np.zeros((2160, 3840, 3), dtype=np.uint8), None, {}, {})
    assert ((int(left), int(ys[0])), (int(right), int(ys[1]))) in strokes
    rendered = _browser({"mode": "overlay", "data": asdict(c), "canvas": {"width": 960, "height": 540}})
    tape = next(s["points"] for s in rendered["strokes"] if s["color"] == "rgba(255, 255, 255, 0.95)")
    np.testing.assert_allclose(tape, np.asarray([[left, ys[0]], [right, ys[1]]]) / 4)
