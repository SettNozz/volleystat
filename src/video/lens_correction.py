"""Preview and prepare a separate rectilinear video using one fisheye map.

The adjustable equidistant model is a manual starting point for Wide footage,
not factory calibration for a particular GoPro or HyperSmooth setting. All
downstream coordinates belong to the prepared video, never to the raw source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import cv2
import numpy as np

VERSION = "fisheye_rectification_v1"
DEFAULT_DIRECTORY = Path(__file__).resolve().parents[2] / "data" / "rectified_videos"


@dataclass(frozen=True)
class LensCorrection:
    enabled: bool = False
    horizontal_fov_deg: float = 130.0
    balance: float = 0.5

    @classmethod
    def from_dict(cls, value=None):
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError("Lens correction must be an object")
        if set(value) - {"enabled", "horizontal_fov_deg", "balance"}:
            raise ValueError("Unknown lens correction setting")
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("Lens correction enabled must be true or false")
        numbers = []
        for key, default, lower, upper in (
            ("horizontal_fov_deg", 130.0, 60, 170), ("balance", 0.5, 0, 1),
        ):
            raw = value.get(key, default)
            if isinstance(raw, bool):
                raise ValueError(f"{key} must be a number")
            try:
                number = float(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{key} must be a finite number") from exc
            if not math.isfinite(number) or not lower <= number <= upper:
                raise ValueError(f"{key} must be between {lower} and {upper}")
            numbers.append(number)
        return cls(enabled, *numbers)

    def to_dict(self):
        return asdict(self)


@lru_cache(maxsize=2)
def correction_maps(width: int, height: int, profile: LensCorrection):
    """Build full-resolution maps once; previews and export share these maps."""
    if width < 2 or height < 2:
        raise ValueError("Lens correction needs an image at least 2 by 2 pixels")
    focal = width / math.radians(profile.horizontal_fov_deg)
    camera = np.array([[focal, 0, (width - 1) / 2],
                       [0, focal, (height - 1) / 2], [0, 0, 1]], dtype=np.float64)
    distortion = np.zeros((4, 1), dtype=np.float64)
    projection = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        camera, distortion, (width, height), np.eye(3), balance=profile.balance,
    )
    return cv2.fisheye.initUndistortRectifyMap(
        camera, distortion, np.eye(3), projection, (width, height), cv2.CV_32FC1,
    )


def rectify_frame(frame, profile: LensCorrection):
    if not profile.enabled:
        return frame
    height, width = frame.shape[:2]
    map_x, map_y = correction_maps(width, height, profile)
    return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def _run_options():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


def _probe(path):
    command = ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    try:
        result = subprocess.run(command, capture_output=True, check=True, **_run_options())
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode(errors="replace")[-1500:]
        raise ValueError("Could not inspect video: " + detail) from exc
    return json.loads(result.stdout)


def _encoder(requested):
    if requested not in {"auto", "h264_nvenc", "libx264"}:
        raise ValueError("Encoder must be auto, h264_nvenc or libx264")
    if requested != "auto":
        return requested
    command = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=256x256:r=1",
               "-frames:v", "1", "-an", "-c:v", "h264_nvenc", "-f", "null", "-"]
    try:
        result = subprocess.run(command, capture_output=True, timeout=15, **_run_options())
        if result.returncode == 0:
            return "h264_nvenc"
    except subprocess.TimeoutExpired:
        pass
    return "libx264"


def _rectified_frames(cap, first, profile):
    """Overlap ordered decoding, remapping and encoder writes with two frames."""
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="lens_prepare") as workers:
        decoded = workers.submit(cap.read)
        corrected = workers.submit(rectify_frame, first, profile)
        while True:
            frame = corrected.result()
            ok, following = decoded.result()
            if ok:
                decoded = workers.submit(cap.read)
                corrected = workers.submit(rectify_frame, following, profile)
            yield frame
            if not ok:
                break


def prepare_video(source, profile: LensCorrection, *, directory=DEFAULT_DIRECTORY,
                  encoder="auto", progress=None):
    """Return a verified, reusable derived video and its provenance record.

    Does not run detectors, update source annotations, or overwrite existing
    artifacts. Changing source bytes or map settings creates a separate path.
    """
    source = Path(source).resolve()
    if not source.is_file():
        raise ValueError(f"Video not found: {source}")
    if not profile.enabled:
        return source, {}
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise ValueError("Lens correction requires ffmpeg and ffprobe on PATH")
    if progress:
        progress({"stage": "lens_correction", "progress": 0,
                  "message": "Checking the Wide video preparation..."})
    source_stat = source.stat()
    source_sha = _sha(source)
    identity = {"version": VERSION, "source_sha256": source_sha,
                "profile": profile.to_dict(), "opencv": cv2.__version__, "encoder": encoder}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    directory = Path(directory).resolve()
    target = directory / key
    video = target / f"{source.stem}.mp4"
    metadata = target / "lens_correction.json"
    if target.exists():
        if metadata.is_file() and video.is_file():
            record = json.loads(metadata.read_text(encoding="utf-8"))
            if record.get("identity") == identity and record.get("output_sha256") == _sha(video):
                return video, {**record, "reused": True}
        raise ValueError(f"Incomplete or changed corrected video; use another output directory: {target}")

    info = _probe(source)
    stream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if stream is None:
        raise ValueError("The source contains no video stream")
    try:
        fps = Fraction(stream["r_frame_rate"])
        average = Fraction(stream.get("avg_frame_rate", stream["r_frame_rate"]))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Could not read a constant video frame rate") from exc
    if fps <= 0 or average <= 0 or abs(float(average / fps) - 1) > .001:
        raise ValueError("Wide correction requires constant-frame-rate footage; normalize timing first")
    selected_encoder = _encoder(encoder)
    cap = cv2.VideoCapture(str(source))
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Could not read the source video")
    height, width = first.shape[:2]
    if width % 2 or height % 2:
        cap.release()
        raise ValueError("Wide correction requires even video dimensions for H.264 playback")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # Each preparation owns its staging directory. Publish only after validation.
        staging = Path(tempfile.mkdtemp(prefix=".rectify_", dir=directory))
    except OSError:
        cap.release()
        raise
    partial = staging / video.name
    audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), {})
    audio_options = ["-c:a", "copy"] if audio.get("codec_name") in {
        "aac", "mp3", "opus", "ac3", "eac3", "alac",
    } else ["-c:a", "aac", "-b:a", "192k"]
    command = ["ffmpeg", "-hide_banner", "-nostdin", "-n", "-v", "error",
               "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
               "-framerate", str(fps), "-i", "pipe:0", "-i", str(source),
               "-map", "0:v:0", "-map", "1:a:0?", "-c:v", selected_encoder]
    command += (["-preset", "p6", "-cq", "16", "-b:v", "0"] if selected_encoder == "h264_nvenc"
                else ["-preset", "medium", "-crf", "16"])
    command += ["-pix_fmt", "yuv420p", *audio_options, "-movflags", "+faststart", str(partial)]
    count = 0
    process = None
    started = published = time.monotonic()
    try:
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=errors, **_run_options())
            try:
                with closing(_rectified_frames(cap, first, profile)) as frames:
                    for frame in frames:
                        process.stdin.write(frame.tobytes())
                        count += 1
                        now = time.monotonic()
                        if progress and now - published >= .5:
                            rate = count / max(now - started, .001)
                            percent = min(99, 100 * count / max(total, count))
                            progress({"stage": "lens_correction", "progress": percent,
                                      "video_percent": percent, "frame_current": count, "frame_total": total,
                                      "processing_fps": rate, "eta_sec": max(0, total - count) / rate,
                                      "message": "Straightening Wide footage before analysis..."})
                            published = now
            except (BrokenPipeError, OSError) as exc:
                process.stdin.close()
                process.wait()
                errors.seek(0)
                detail = errors.read().decode(errors="replace")[-2000:]
                raise ValueError("Corrected video encoding failed: " + detail) from exc
            finally:
                if not process.stdin.closed:
                    process.stdin.close()
            if process.wait() != 0:
                errors.seek(0)
                raise ValueError("Corrected video encoding failed: " + errors.read().decode(errors="replace")[-2000:])
        if count == 0:
            raise ValueError("No frames were corrected")
        output_info = _probe(partial)
        output_stream = next(s for s in output_info["streams"] if s["codec_type"] == "video")
        if int(output_stream.get("nb_frames", 0)) != count:
            raise ValueError("Corrected video frame count does not match the decoded source")
        # A truncated decode must not silently become a successful full-video job.
        duration = float(info.get("format", {}).get("duration", count / float(fps)))
        if abs(count / float(fps) - duration) > max(.1, 2 / float(fps)):
            raise ValueError("Source duration and decoded frames disagree; check for variable timing or decode errors")
        if (source.stat().st_size, source.stat().st_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
            raise ValueError("The source changed during correction")
        record = {"identity": identity, "source_video_path": str(source), "source_sha256": source_sha,
                  "output_sha256": _sha(partial), "output_video_path": str(video),
                  "coordinate_space": "rectified_image_pixels", "frame_width": width, "frame_height": height,
                  "frames": count, "fps": str(fps), "encoder": selected_encoder,
                  "model": "manual_equidistant_fisheye", "factory_calibrated": False}
        (staging / metadata.name).write_text(json.dumps(record, indent=2), encoding="utf-8")
        # rename refuses an existing nonempty target; concurrent runs cannot replace it.
        staging.rename(target)
        return video, {**record, "reused": False}
    finally:
        cap.release()
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if staging.exists():
            for owned in (partial, staging / metadata.name):
                owned.unlink(missing_ok=True)
            staging.rmdir()
