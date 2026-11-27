"""Geometry, source preservation and corrected-video cache integration."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from src.video.lens_correction import LensCorrection, prepare_video, rectify_frame


@pytest.mark.parametrize("value", [True, [], {"enabled": "true"}, {"balance": float("nan")},
                                  {"balance": 2}, {"horizontal_fov_deg": 180},
                                  {"horizontal_fov_deg": True}, {"unknown": 1}])
def test_invalid_profile_is_rejected(value):
    with pytest.raises(ValueError):
        LensCorrection.from_dict(value)


def test_disabled_correction_does_not_change_pixels():
    frame = np.random.default_rng(4).integers(0, 256, (90, 160, 3), dtype=np.uint8)
    assert rectify_frame(frame, LensCorrection()) is frame


def test_fisheye_projection_of_a_straight_line_becomes_straight():
    # Project a real straight 3D line through an equidistant fisheye camera.
    width, height = 640, 360
    focal = width / np.deg2rad(130)
    camera = np.array([[focal, 0, (width - 1) / 2],
                       [0, focal, (height - 1) / 2], [0, 0, 1]], np.float64)
    points = np.array([[x, .35, 1] for x in np.linspace(-1.3, 1.3, 300)], np.float64)
    projected, _ = cv2.fisheye.projectPoints(points[None], np.zeros(3), np.zeros(3),
                                            camera, np.zeros(4))
    image = np.zeros((height, width, 3), np.uint8)
    cv2.polylines(image, [projected.reshape(-1, 2).round().astype(np.int32)], False, (255, 255, 255), 1)
    corrected = rectify_frame(image, LensCorrection(enabled=True))
    before_y, _ = np.where(image[:, :, 0] > 128)
    after_y, _ = np.where(corrected[:, :, 0] > 128)
    assert before_y.std() > 4
    assert len(after_y) > 100
    assert after_y.std() < 1.2
    assert corrected.shape == image.shape


@pytest.fixture
def source_video(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are required")
    source = tmp_path / "wide.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i",
                    "testsrc2=size=160x96:rate=12", "-f", "lavfi", "-i",
                    "sine=frequency=880:sample_rate=48000", "-t", "1", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(source)], check=True, capture_output=True)
    return source


def test_preparation_preserves_source_audio_frames_and_reuses_verified_output(source_video, tmp_path, monkeypatch):
    before = source_video.read_bytes()
    profile = LensCorrection(enabled=True)
    video, record = prepare_video(source_video, profile, directory=tmp_path / "prepared", encoder="libx264")
    assert source_video.read_bytes() == before
    assert video != source_video
    assert record["frames"] == 12
    assert record["fps"] == "12"
    assert record["factory_calibrated"] is False
    assert record["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert record["coordinate_space"] == "rectified_image_pixels"
    info = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)]))
    assert {s["codec_type"] for s in info["streams"]} == {"video", "audio"}
    assert info["streams"][0]["nb_frames"] == "12"

    cap = cv2.VideoCapture(str(source_video))
    ok, raw = cap.read()
    cap.release()
    assert ok
    cap = cv2.VideoCapture(str(video))
    ok, actual = cap.read()
    cap.release()
    assert ok
    expected = rectify_frame(raw, profile)
    assert np.abs(actual.astype(float) - expected).mean() < 7

    def unexpected_decode(*args):
        raise AssertionError("Verified repeat must not decode or re-encode the video")

    monkeypatch.setattr(cv2, "VideoCapture", unexpected_decode)
    reused, second = prepare_video(source_video, profile, directory=tmp_path / "prepared", encoder="libx264")
    assert reused == video
    assert second["reused"] is True
    video.write_bytes(video.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed corrected video"):
        prepare_video(source_video, profile, directory=tmp_path / "prepared", encoder="libx264")


def test_changed_lens_settings_create_a_separate_video(source_video, tmp_path):
    directory = tmp_path / "prepared"
    first, _ = prepare_video(source_video, LensCorrection(enabled=True), directory=directory, encoder="libx264")
    old = first.read_bytes()
    second, _ = prepare_video(source_video, LensCorrection(True, 115, .5), directory=directory, encoder="libx264")
    assert first != second
    assert first.read_bytes() == old


def test_variable_frame_rate_is_refused_before_creating_output(source_video, tmp_path, monkeypatch):
    from src.video import lens_correction
    monkeypatch.setattr(lens_correction, "_probe", lambda p: {
        "streams": [{"codec_type": "video", "r_frame_rate": "30", "avg_frame_rate": "24"}],
    })
    with pytest.raises(ValueError, match="constant-frame-rate"):
        prepare_video(source_video, LensCorrection(enabled=True), directory=tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_auto_encoder_probe_uses_dimensions_supported_by_nvenc(monkeypatch):
    from src.video import lens_correction
    commands = []

    def probe(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", probe)
    assert lens_correction._encoder("auto") == "h264_nvenc"
    assert "color=s=256x256:r=1" in commands[0]


def test_preparation_workers_preserve_frame_order_and_decode_errors():
    from contextlib import closing
    from src.video.lens_correction import _rectified_frames

    class Capture:
        index = 0

        def read(self):
            self.index += 1
            if self.index == 5:
                raise ValueError("decode failed")
            return True, np.full((2, 2, 3), self.index, np.uint8)

    observed = []
    with pytest.raises(ValueError, match="decode failed"):
        with closing(_rectified_frames(Capture(), np.zeros((2, 2, 3), np.uint8), LensCorrection())) as frames:
            for frame in frames:
                observed.append(int(frame[0, 0, 0]))
    assert observed == [0, 1, 2, 3]
