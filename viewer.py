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
CORS(app)

RESULTS_DIR = Path("results/viewer")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Track running jobs
jobs: dict = {}


def _run_analysis(job_id: str, video_path: str, skip_frames: int):
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
        )
        jobs[job_id]["message"] = "Analyzing video..."
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
    skip_frames = int(data.get("skip_frames", 1))

    if not video_path or not Path(video_path).exists():
        return jsonify({"error": f"Video not found: {video_path}"}), 400

    job_id = f"job_{int(time.time())}"
    t = threading.Thread(target=_run_analysis, args=(job_id, video_path, skip_frames), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    return jsonify(jobs.get(job_id, {"status": "not_found"}))


@app.route("/results/<job_id>")
def results(job_id):
    job = jobs.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Results not ready"}), 404

    with open(job["stats_path"]) as f:
        stats = json.load(f)
    return render_template("viewer.html", job=job, stats=stats)


@app.route("/api/annotations/<job_id>")
def annotations(job_id):
    job = jobs.get(job_id)
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
    job = jobs.get(job_id)
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
