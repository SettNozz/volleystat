#!/usr/bin/env python3
"""
Volleyball Analytics Viewer

Flask web app for visual validation of the analysis pipeline.
Upload a video → analyze → view annotated results with interactive overlays.

Usage:
    python viewer.py
    # Open http://localhost:5050 in browser
"""

import json
import math
import mimetypes
import os
import sys
import threading
import time
import urllib.parse
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, Response
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent))

from src.tracking.attack_heatmap import HEATMAP_VERSION
from src.tracking.court_setups import (
    CourtSetup,
    CourtSetupError,
    CourtSetupStore,
    court_guide_lines,
    homography_from_corners,
    scale_corners,
    validate_corners,
)
from src.utils.path_utils import local_path_from_uri

app = Flask(__name__, template_folder="viewer/templates", static_folder="viewer/static")
# Jinja caches compiled templates in memory, so UI edits need a server restart
# unless auto-reload is on. The viewer is a local dev tool — always reload.
app.config["TEMPLATES_AUTO_RELOAD"] = True
CORS(app)

# Server can be launched from a shortcut or a different shell directory.
# Persisted job recovery must therefore never depend on the process CWD.
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "viewer"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
COURT_SETUPS_PATH = PROJECT_ROOT / "configs" / "court_setups.json"


def _setup_store() -> CourtSetupStore:
    return CourtSetupStore(COURT_SETUPS_PATH)

# Track running jobs
jobs: dict = {}

_LIFECYCLE_COUNTERS = (
    "service_attempts",
    "confirmed_serves",
    "playable_rallies",
    "aces_no_reception",
    "reception_failed",
    "serve_faults",
)


def _prepare_viewer_stats(stats: dict) -> dict:
    """Add a small, stable presentation contract without reclassifying events.

    New analysis results provide canonical rally lifecycle metadata.  Legacy
    results predate that schema, so the viewer keeps their original rally list
    and total instead of guessing service outcomes from noisy contact events.
    """
    prepared = dict(stats)
    prepared["rallies"] = list(stats.get("rallies") or [])
    prepared["service_attempts"] = list(stats.get("service_attempts") or [])
    prepared.setdefault("all_events", [])

    lifecycle = stats.get("rally_lifecycle")
    lifecycle_available = isinstance(lifecycle, dict)
    if lifecycle_available:
        counters = {
            key: int(lifecycle.get(key, 0) or 0)
            for key in _LIFECYCLE_COUNTERS
        }
        prepared["rallies"] = [
            rally for rally in prepared["rallies"]
            if isinstance(rally, dict)
            and isinstance(rally.get("lifecycle"), dict)
            and rally["lifecycle"].get("status") == "playable_rally"
            and rally["lifecycle"].get("is_playable") is True
        ]
        # The canonical lifecycle counter is the user-facing rally total.
        prepared["total_rallies"] = counters["playable_rallies"]
    else:
        legacy_total = int(
            stats.get("total_rallies", len(prepared["rallies"])) or 0
        )
        counters = {
            "service_attempts": None,
            "confirmed_serves": None,
            "playable_rallies": legacy_total,
            "aces_no_reception": None,
            "reception_failed": None,
            "serve_faults": None,
        }
        prepared.setdefault("total_rallies", legacy_total)

    prepared["viewer_lifecycle"] = {
        "available": lifecycle_available,
        **counters,
    }
    return prepared


def _resolve_video_path(value: str) -> Path:
    """Resolve regular paths and file URIs independently of the server CWD."""
    path = local_path_from_uri(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _restore_completed_job(job_id: str):
    """Restore a persisted viewer result after the Flask process restarts."""
    if not job_id.startswith("job_"):
        return None
    output_dir = RESULTS_DIR / job_id
    stats_files = sorted(output_dir.glob("*_stats.json"))
    if not stats_files:
        return None

    stats_path = stats_files[0]
    stem = stats_path.name.removesuffix("_stats.json")
    ann_path = output_dir / f"{stem}_annotations.json"
    video_path = output_dir / f"{stem}_analyzed.mp4"
    if not ann_path.exists():
        return None

    original_video = ""
    try:
        with open(ann_path) as f:
            original_video = json.load(f).get("original_video_path", "")
    except (OSError, json.JSONDecodeError):
        pass
    playback_path = (
        video_path if video_path.exists()
        else _resolve_video_path(original_video) if original_video
        else None
    )
    if playback_path is None or not playback_path.exists():
        return None

    job = {
        "status": "done",
        "progress": 100,
        "message": "Restored from disk",
        "stats_path": str(stats_path),
        "annotations_path": str(ann_path),
        "video_path": str(playback_path),
        "video_stem": stem,
        "job_id": job_id,
        "original_video_path": original_video,
        "original_video_path_url": urllib.parse.quote(original_video, safe=""),
    }
    jobs[job_id] = job
    return job


def _get_job(job_id: str):
    return jobs.get(job_id) or _restore_completed_job(job_id)


def _run_analysis(
    job_id: str,
    video_path: str,
    skip_frames: int,
    refresh_net_floor: bool = True,
    ball_batch_size: int = 8,
    net_height_m: float = 2.34,
):
    """Run analysis in background thread."""
    jobs[job_id] = {"status": "running", "progress": 0, "message": "Starting..."}
    try:
        from src.pipeline.game_analyzer import GameAnalyzer
        output_dir = RESULTS_DIR / job_id
        output_dir.mkdir(exist_ok=True)

        def update_progress(update: dict) -> None:
            """Publish analyzer progress atomically for the polling endpoint."""
            current = dict(jobs.get(job_id) or {})
            if current.get("status") == "done":
                return
            current.update(update)
            current["status"] = "running"
            jobs[job_id] = current

        analyzer = GameAnalyzer(
            # The results page draws overlays from annotations on top of the
            # original video. Encoding a second 4K MP4 is redundant and nearly
            # doubles wall time.
            output_video=False,
            skip_frames=skip_frames,
            pose_imgsz=1280,
            max_players_per_team=6,
            refresh_net_floor=refresh_net_floor,
            ball_batch_size=ball_batch_size,
            net_height_m=net_height_m,
            progress_callback=update_progress,
        )
        jobs[job_id]["message"] = (
            f"GPU batch {ball_batch_size}: skeletons + SAM3 net posts + heatmap..."
        )
        stats = analyzer.process_video(
            video_path=video_path,
            output_dir=str(output_dir),
        )
        jobs[job_id] = {
            "status": "done",
            "progress": 100,
            "message": "Done",
            "stats_path": str(output_dir / f"{Path(video_path).stem}_stats.json"),
            "annotations_path": str(output_dir / f"{Path(video_path).stem}_annotations.json"),
            "video_path": video_path,
            "video_stem": Path(video_path).stem,
            "job_id": job_id,
            "original_video_path": video_path,
            "original_video_path_url": urllib.parse.quote(str(video_path), safe=""),
        }
    except Exception as e:
        jobs[job_id] = {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/load")
def load_existing():
    """Load pre-existing analysis results without re-running analysis.

    Usage: /load?video=original_images_video&results_dir=results/test_video
    """
    stem = request.args.get("video", "")
    results_dir = Path(request.args.get("results_dir", "results/viewer"))
    orig_video = request.args.get("original_video", "")

    stats_path = results_dir / f"{stem}_stats.json"
    ann_path = results_dir / f"{stem}_annotations.json"
    video_path = results_dir / f"{stem}_analyzed.mp4"

    if not stats_path.exists():
        return f"Stats not found: {stats_path}", 404

    # Auto-recover original video path from annotations JSON if not in URL
    if not orig_video and ann_path.exists():
        try:
            with open(ann_path) as f:
                ann = json.load(f)
            orig_video = ann.get("original_video_path", "")
        except Exception:
            pass

    job_id = f"preloaded_{stem}"
    jobs[job_id] = {
        "status": "done",
        "progress": 100,
        "message": "Pre-loaded",
        "stats_path": str(stats_path),
        "annotations_path": str(ann_path),
        "video_path": str(video_path),
        "video_stem": stem,
        "job_id": job_id,
        "original_video_path": orig_video,
        "original_video_path_url": urllib.parse.quote(orig_video, safe=""),
    }
    from flask import redirect
    return redirect(f"/results/{job_id}")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    video_path_raw = str(data.get("video_path", "")).strip()
    skip_frames = int(data.get("skip_frames", 0))
    ball_batch_size = max(1, min(16, int(data.get("ball_batch_size", 8))))
    net_profile = str(data.get("net_profile", "mixed")).lower()
    net_height_m = 2.43 if net_profile == "men" else 2.34
    if data.get("net_height_m") is not None:
        net_height_m = float(data["net_height_m"])
    if not 2.0 <= net_height_m <= 2.6:
        return jsonify({"error": "net_height_m must be between 2.0 and 2.6"}), 400
    # A viewer upload is normally a new recording. Re-detect its two physical
    # net-post bases instead of reusing endpoints from another camera angle.
    refresh_net_floor = bool(data.get("refresh_net_floor", True))

    video_path = _resolve_video_path(video_path_raw) if video_path_raw else None
    if video_path is None or not video_path.is_file():
        return jsonify({"error": f"Video not found: {video_path_raw}"}), 400

    job_id = f"job_{int(time.time())}"
    t = threading.Thread(
        target=_run_analysis,
        args=(
            job_id, str(video_path), skip_frames, refresh_net_floor,
            ball_batch_size, net_height_m,
        ),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    return jsonify(_get_job(job_id) or {"status": "not_found"})


@app.route("/results/<job_id>")
def results(job_id):
    job = _get_job(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Results not ready"}), 404

    with open(job["stats_path"]) as f:
        stats = _prepare_viewer_stats(json.load(f))
    return render_template("viewer.html", job=job, stats=stats)


@app.route("/api/annotations/<job_id>")
def annotations(job_id):
    job = _get_job(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    return send_file(job["annotations_path"], mimetype="application/json")


def _read_annotations(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _write_annotations(path: str, payload: dict) -> None:
    """Atomically persist viewer-only calibration corrections."""
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temporary.replace(target)


def _validated_corners(value, width: float, height: float) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("court_corners must contain exactly four [x, y] points")
    corners: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("each court corner must be an [x, y] pair")
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("court corners must be finite numbers")
        # A wide lens routinely pushes a real court corner outside the frame,
        # so only obviously broken coordinates are rejected here.
        if not (-width <= x <= width * 2 and -height <= y <= height * 2):
            raise ValueError("court corner lies implausibly far outside the video")
        corners.append([round(x, 2), round(y, 2)])
    return corners


def _annotation_corners(payload: dict) -> list | None:
    """Best available court corners for a result, in its own pixel space.

    Results analysed before the viewer stored corners fall back to the shared
    calibration, rescaled — that is exactly the geometry the analysis used.
    """
    corners = payload.get("court_corners")
    if corners and len(corners) == 4:
        return [[float(point[0]), float(point[1])] for point in corners]

    calibration_path = PROJECT_ROOT / "court_calibration.json"
    if not calibration_path.exists():
        return None
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    corners = calibration.get("court_corners")
    if not corners or len(corners) != 4:
        return None
    return [
        list(point) for point in scale_corners(
            corners,
            (calibration.get("frame_width", 3840), calibration.get("frame_height", 2160)),
            (payload.get("frame_width", 1280), payload.get("frame_height", 720)),
        )
    ]


def _projector_for(payload: dict, corners: list):
    """Court projector for a result from its (possibly corrected) corners."""
    from src.detection.player_pose import CourtProjector

    homography = homography_from_corners(corners)
    floor_left = payload.get("net_floor_left")
    floor_right = payload.get("net_floor_right")
    floor_line = (
        (tuple(floor_left), tuple(floor_right))
        if floor_left and floor_right else None
    )
    tape_band = None
    if floor_line and payload.get("net_tape_top") and payload.get("net_tape_bottom"):
        x1, x2 = sorted((floor_left[0], floor_right[0]))
        tape_band = (x1, x2, payload["net_tape_top"], payload["net_tape_bottom"])
    return CourtProjector(
        homography, net_floor_line=floor_line, net_tape_band=tape_band,
    ), homography


def _recompute_attack_heatmap(job: dict, payload: dict) -> dict:
    """Rebuild the attack map from stored annotations and the current court.

    Re-running the analyzer takes tens of minutes; every input the heatmap
    needs (ball, poses, events) is already in the annotations, so a corrected
    court only requires this pass.
    """
    from src.tracking.attack_heatmap import AttackHeatmap

    corners = _annotation_corners(payload)
    if corners is None:
        raise ValueError("no court corners available — draw the court first")
    projector, _ = _projector_for(payload, corners)

    events = None
    try:
        with open(job["stats_path"], encoding="utf-8") as handle:
            events = (json.load(handle) or {}).get("all_events")
    except (OSError, json.JSONDecodeError, KeyError):
        events = None

    # The post bases mark the physical ends of the net; net_x_left/right are
    # calibration clicks that can sit well inside them.
    floor_left = payload.get("net_floor_left")
    floor_right = payload.get("net_floor_right")
    net_x_left = floor_left[0] if floor_left else payload.get("net_x_left")
    net_x_right = floor_right[0] if floor_right else payload.get("net_x_right")

    return AttackHeatmap(payload.get("fps") or 30.0).analyze(
        payload.get("frames") or [],
        projector=projector,
        events=events,
        net_y=payload.get("net_y"),
        net_x_left=net_x_left,
        net_x_right=net_x_right,
    )


def _court_state(job: dict, payload: dict) -> dict:
    corners = _annotation_corners(payload)
    guides = None
    error = None
    if corners is not None:
        try:
            guides = court_guide_lines(homography_from_corners(corners))
        except CourtSetupError as exc:
            error = str(exc)
    return {
        "court_corners": corners,
        "guides": guides,
        "guides_error": error,
        "net_y": payload.get("net_y"),
        "net_x_left": payload.get("net_x_left"),
        "net_x_right": payload.get("net_x_right"),
        "frame_width": payload.get("frame_width"),
        "frame_height": payload.get("frame_height"),
        "court_setup_id": payload.get("court_setup_id"),
        "court_setup_name": payload.get("court_setup_name"),
        "manual_court_correction": payload.get("manual_court_correction"),
        "heatmap_version": (payload.get("attack_heatmap") or {}).get("version"),
        "current_heatmap_version": HEATMAP_VERSION,
        "setups": [setup.to_dict() for setup in _setup_store().load()],
    }


@app.route("/api/court/<job_id>", methods=["GET", "PUT"])
def court_correction(job_id):
    """Read or save a per-result court correction from the viewer.

    A correction belongs to one recorded game: the camera is re-mounted
    between games, so this never silently rewrites the shared
    ``court_calibration.json``.  Saving it under a name (``save_as_setup``)
    makes it reusable for other games shot from the same position.
    """
    job = _get_job(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    payload = _read_annotations(job["annotations_path"])
    if request.method == "GET":
        return jsonify(_court_state(job, payload))

    update = request.get_json(silent=True) or {}
    width = float(payload.get("frame_width") or 1280)
    height = float(payload.get("frame_height") or 720)
    try:
        payload["court_corners"] = _validated_corners(
            update.get("court_corners"), width, height,
        )
        validate_corners(payload["court_corners"])
        for key in ("net_y", "net_x_left", "net_x_right"):
            if key not in update:
                continue
            number = float(update[key])
            if not math.isfinite(number):
                raise ValueError(f"{key} must be finite")
            payload[key] = round(number, 2)

        setup_name = str(update.get("save_as_setup") or "").strip()
        if setup_name:
            setup = _setup_store().save(CourtSetup(
                name=setup_name,
                court_corners=payload["court_corners"],
                frame_width=width,
                frame_height=height,
                id=str(update.get("court_setup_id") or "") or uuid.uuid4().hex[:12],
                net_y=payload.get("net_y"),
                net_x_left=payload.get("net_x_left"),
                net_x_right=payload.get("net_x_right"),
            ))
            payload["court_setup_id"] = setup.id
            payload["court_setup_name"] = setup.name

        payload["manual_court_correction"] = {
            "saved_at_unix": round(time.time(), 3),
            "source": "viewer",
        }
        payload["attack_heatmap"] = _recompute_attack_heatmap(job, payload)
        _write_annotations(job["annotations_path"], payload)
    except (CourtSetupError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    state = _court_state(job, payload)
    state["attack_heatmap"] = payload["attack_heatmap"]
    return jsonify(state)


@app.route("/api/court/<job_id>/recompute", methods=["POST"])
def recompute_court_stats(job_id):
    """Rebuild the attack map for an already-analysed result.

    Older results carry an earlier heatmap schema with no measured shot
    directions.  This replaces it in place rather than making the user
    re-analyse the video.
    """
    job = _get_job(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    payload = _read_annotations(job["annotations_path"])
    try:
        payload["attack_heatmap"] = _recompute_attack_heatmap(job, payload)
        if payload.get("court_corners") is None:
            payload["court_corners"] = _annotation_corners(payload)
        _write_annotations(job["annotations_path"], payload)
    except (CourtSetupError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"attack_heatmap": payload["attack_heatmap"]})


@app.route("/api/court_setups", methods=["GET", "POST"])
def court_setups():
    store = _setup_store()
    if request.method == "GET":
        return jsonify({"setups": [setup.to_dict() for setup in store.load()]})

    data = request.get_json(silent=True) or {}
    try:
        setup = store.save(CourtSetup(
            name=data.get("name", ""),
            court_corners=data.get("court_corners"),
            frame_width=data.get("frame_width", 0),
            frame_height=data.get("frame_height", 0),
            id=str(data.get("id") or "") or uuid.uuid4().hex[:12],
            net_y=data.get("net_y"),
            net_x_left=data.get("net_x_left"),
            net_x_right=data.get("net_x_right"),
            note=str(data.get("note") or ""),
        ))
    except (CourtSetupError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(setup.to_dict())


@app.route("/api/court_setups/<setup_id>", methods=["DELETE"])
def delete_court_setup(setup_id):
    if not _setup_store().delete(setup_id):
        return jsonify({"error": "Setup not found"}), 404
    return jsonify({"deleted": setup_id})


def _stream_video(video_path: str):
    """Stream a video file with HTTP Range support for HTML5 player."""
    p = Path(video_path)
    if not p.exists():
        return ("Video not found", 404)

    mimetype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    file_size = os.path.getsize(video_path)
    range_header = request.headers.get("Range")

    if range_header:
        byte_start, byte_end = 0, file_size - 1
        match = range_header.replace("bytes=", "").split("-")
        byte_start = int(match[0])
        if match[1]:
            byte_end = int(match[1])
        chunk_size = byte_end - byte_start + 1

        def generate():
            with open(video_path, "rb") as f:
                f.seek(byte_start)
                remaining = chunk_size
                while remaining > 0:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {byte_start}-{byte_end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": chunk_size,
            "Content-Type": mimetype,
        }
        return Response(generate(), 206, headers=headers)

    return send_file(video_path, mimetype=mimetype)


@app.route("/api/video/<job_id>")
def video(job_id):
    job = _get_job(job_id)
    if not job or job["status"] != "done":
        return ("Not ready", 404)
    return _stream_video(job["video_path"])


@app.route("/api/original_video")
def original_video():
    """Serve the original (unannotated) video for canvas overlay mode."""
    return _stream_video(request.args.get("path", ""))


if __name__ == "__main__":
    print("\nVolleyball Analytics Viewer")
    print("Open http://localhost:5050 in your browser\n")
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
