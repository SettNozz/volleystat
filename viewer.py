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
import os
import sys
import threading
import time
import urllib.parse
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, Response
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__, template_folder="viewer/templates", static_folder="viewer/static")
# Jinja caches compiled templates in memory, so UI edits need a server restart
# unless auto-reload is on. The viewer is a local dev tool — always reload.
app.config["TEMPLATES_AUTO_RELOAD"] = True
CORS(app)

# Server can be launched from a shortcut or a different shell directory.
# Persisted job recovery must therefore never depend on the process CWD.
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "viewer"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Track running jobs
jobs: dict = {}


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
    if not ann_path.exists() or not video_path.exists():
        return None

    original_video = ""
    try:
        with open(ann_path) as f:
            original_video = json.load(f).get("original_video_path", "")
    except (OSError, json.JSONDecodeError):
        pass

    job = {
        "status": "done",
        "progress": 100,
        "message": "Restored from disk",
        "stats_path": str(stats_path),
        "annotations_path": str(ann_path),
        "video_path": str(video_path),
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
):
    """Run analysis in background thread."""
    jobs[job_id] = {"status": "running", "progress": 0, "message": "Starting..."}
    try:
        from src.pipeline.game_analyzer import GameAnalyzer
        output_dir = RESULTS_DIR / job_id
        output_dir.mkdir(exist_ok=True)

        # Patch progress into jobs dict during analysis
        # (GameAnalyzer prints progress — we capture state changes)
        analyzer = GameAnalyzer(
            output_video=True,
            skip_frames=skip_frames,
            pose_imgsz=1280,
            max_players_per_team=6,
            refresh_net_floor=refresh_net_floor,
        )
        jobs[job_id]["message"] = "Full workflow: skeletons + SAM3 net posts + heatmap..."
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
            "video_path": str(output_dir / f"{Path(video_path).stem}_analyzed.mp4"),
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
    data = request.json
    video_path = data.get("video_path", "").strip()
    skip_frames = int(data.get("skip_frames", 0))
    # A viewer upload is normally a new recording. Re-detect its two physical
    # net-post bases instead of reusing endpoints from another camera angle.
    refresh_net_floor = bool(data.get("refresh_net_floor", True))

    if not video_path or not Path(video_path).exists():
        return jsonify({"error": f"Video not found: {video_path}"}), 400

    job_id = f"job_{int(time.time())}"
    t = threading.Thread(
        target=_run_analysis,
        args=(job_id, video_path, skip_frames, refresh_net_floor),
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
        stats = json.load(f)
    return render_template("viewer.html", job=job, stats=stats)


@app.route("/api/annotations/<job_id>")
def annotations(job_id):
    job = _get_job(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    return send_file(job["annotations_path"], mimetype="application/json")


def _stream_video(video_path: str):
    """Stream a video file with HTTP Range support for HTML5 player."""
    p = Path(video_path)
    if not p.exists():
        return ("Video not found", 404)

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
            "Content-Type": "video/mp4",
        }
        return Response(generate(), 206, headers=headers)

    return send_file(video_path, mimetype="video/mp4")


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
