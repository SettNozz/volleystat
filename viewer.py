#!/usr/bin/env python3
"""
Volleyball Analytics Viewer

Flask web app for visual validation of the analysis pipeline.
Upload a video → analyze → view annotated results with interactive overlays.

Usage:
    python viewer.py
    # Open http://localhost:5050 in browser
"""

import base64
import hashlib
import json
import math
import os
import sys
import threading
import time
import urllib.parse
import uuid
from datetime import datetime, timezone
from collections import OrderedDict
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file, Response, url_for
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
from src.models.model_registry import fallback_path, load_config as load_registry_config
from src.tracking.player_contacts import apply_reviews, source_fingerprint
from src.tracking.player_match_statistics import build_player_match_statistics
from src.tracking.statistics_reconciliation import statistic_evidence
from src.tracking.player_statistics_excel import (
    PLAYER_POSITIONS, STATISTICS_VIEWS, export_player_statistics, player_statistic_value,
)
from src.viewer_app.player_highlights import HIGHLIGHT_CATEGORIES, player_highlights
from src.viewer_app.statistics_reviews import (
    StatisticsSourceChanged, load_point_reviews, validate_point_review, append_point_review,
)
from src.viewer_app.match_workspace import (
    read_workspace, workspace_path, author_review_path, guard_source, append_record,
    validate_context, project_context, score_projection, context_at,
)
from src.tracking.contact_author_experiments import ExperimentSourceChanged
from src.tracking.volleyball_contact_lab import load_lab
from src.viewer_app.stats import _prepare_viewer_stats
from src.viewer_app.library import _annotation_original_video, _library_summary
from src.viewer_app.calibration_preview import _manual_calibration, _calibration_frame
from src.viewer_app.player_contacts_io import _load_player_contact_inputs, _player_contact_inputs  # noqa: F401 (tests clear its cache)
from src.viewer_app.court_correction import (
    _read_annotations, _write_annotations, _validated_corners, _projector_for,
    _sync_manual_net_geometry,
)
from src.viewer_app.media import _stream_video, _decode_thumbnail, _decode_player_portrait
from src.viewer_app.uploads import UploadError, find_existing_upload, store_upload

app = Flask(__name__, template_folder="viewer/templates", static_folder="viewer/static")
# Jinja caches compiled templates in memory, so UI edits need a server restart
# unless auto-reload is on. The viewer is a local dev tool — always reload.
app.config["TEMPLATES_AUTO_RELOAD"] = True
CORS(app)

# Server can be launched from a shortcut or a different shell directory.
# Persisted job recovery must therefore never depend on the process CWD.
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_DIR = RESULTS_ROOT / "viewer"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
COURT_SETUPS_PATH = PROJECT_ROOT / "configs" / "court_setups.json"
# Videos dropped on the New analysis page (browsers never expose their path).
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
# Viewer uploads run the dense contact spotter, so their events and rallies
# match `analyze_video.py --contact-spotter`. Without the checkpoint the
# analysis falls back to the FSM's events and rallies. The file is the
# `contact_spotter` entry of configs/model_registry.json (pinned, backed up).
CONTACT_SPOTTER_PATH = (
    fallback_path(load_registry_config().get("contact_spotter", {}))
    or PROJECT_ROOT / "models" / "contact_spotter" / "spotter_20261028_v1.pt"
)


def _setup_store() -> CourtSetupStore:
    return CourtSetupStore(COURT_SETUPS_PATH)

# Track running jobs
jobs: dict = {}


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


def _library_entry(directory: Path, stem: str) -> dict | None:
    """Describe one ``<stem>_*`` result set inside ``directory``."""
    stats_path = directory / f"{stem}_stats.json"
    ann_path = directory / f"{stem}_annotations.json"
    annotated_video = directory / f"{stem}_analyzed.mp4"

    present = [p for p in (stats_path, ann_path, annotated_video) if p.is_file()]
    if not present:
        return None

    original_video = _annotation_original_video(ann_path) if ann_path.is_file() else ""
    original_exists = bool(original_video) and _resolve_video_path(original_video).is_file()
    rel_dir = directory.relative_to(PROJECT_ROOT).as_posix()
    job_id = directory.name if directory.parent == RESULTS_DIR else None

    entry = {
        "stem": stem,
        "results_dir": rel_dir,
        "job_id": job_id,
        "modified": max(p.stat().st_mtime for p in present),
        "size_bytes": sum(p.stat().st_size for p in present),
        "has_stats": stats_path.is_file(),
        "has_annotations": ann_path.is_file(),
        "has_annotated_video": annotated_video.is_file(),
        "original_video": original_video,
        "playable": annotated_video.is_file() or original_exists,
        "summary": _library_summary(stats_path) if stats_path.is_file() else None,
    }

    # Both viewer routes read `<stem>_stats.json`, so a stats-less folder is
    # listed for visibility but cannot be opened. Persisted upload jobs restore
    # through /results/<job_id>; anything else is addressed by directory.
    if not entry["has_stats"]:
        entry["open_url"] = None
        entry["blocked_reason"] = f"{stem}_stats.json missing"
    elif job_id and entry["playable"]:
        entry["open_url"] = f"/results/{job_id}"
        entry["blocked_reason"] = None
    else:
        query = urllib.parse.urlencode({"video": stem, "results_dir": rel_dir})
        entry["open_url"] = f"/load?{query}"
        entry["blocked_reason"] = None
    return entry


def _library_index() -> list[dict]:
    """List every analysed result under results/, newest first."""
    if not RESULTS_ROOT.is_dir():
        return []

    entries: list[dict] = []
    seen: set[tuple[Path, str]] = set()
    for suffix in ("_stats.json", "_annotations.json"):
        for path in RESULTS_ROOT.rglob(f"*{suffix}"):
            stem = path.name[: -len(suffix)]
            key = (path.parent, stem)
            if key in seen:
                continue
            seen.add(key)
            entry = _library_entry(path.parent, stem)
            if entry:
                entries.append(entry)

    entries.sort(key=lambda item: item["modified"], reverse=True)
    return entries


def _run_analysis(
    job_id: str,
    video_path: str,
    skip_frames: int,
    refresh_net_floor: bool = True,
    net_height_m: float = 2.34,
    calibration: dict | None = None,
):
    """Run analysis in background thread."""
    jobs[job_id] = {"status": "running", "progress": 0, "message": "Starting..."}
    try:
        from src.pipeline.game_analyzer import GameAnalyzer
        output_dir = RESULTS_DIR / job_id
        output_dir.mkdir(exist_ok=True)
        calibration_options = {}
        if calibration is not None:
            path = output_dir / "court_calibration.json"
            path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
            calibration_options = {
                "calibration_path": str(path.resolve()),
                "enable_net_detection": False,
            }

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
            pose_imgsz=2560,
            max_players_per_team=6,
            refresh_net_floor=refresh_net_floor,
            ball_batch_size=None,
            net_height_m=net_height_m,
            progress_callback=update_progress,
            contact_spotter_path=(
                str(CONTACT_SPOTTER_PATH) if CONTACT_SPOTTER_PATH.is_file() else None
            ),
            **calibration_options,
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


@app.route("/favicon.ico")
def favicon():
    """Browsers ask for /favicon.ico even on pages that link another icon."""
    return send_file(PROJECT_ROOT / "viewer" / "static" / "brand" / "favicon.ico",
                     mimetype="image/vnd.microsoft.icon", max_age=86400)


@app.route("/library")
def library():
    """Browse every analysed result on disk and open one in the viewer."""
    return render_template("library.html")


@app.route("/api/library")
def api_library():
    entries = _library_index()
    return jsonify({
        "root": RESULTS_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "count": len(entries),
        "entries": entries,
    })


@app.route("/load")
def load_existing():
    """Load pre-existing analysis results without re-running analysis.

    Usage: /load?video=original_images_video&results_dir=results/test_video
    """
    stem = request.args.get("video", "")
    results_dir = Path(request.args.get("results_dir", "results/viewer"))
    # The server may be launched from any directory, so a relative results path
    # is anchored to the project instead of the process CWD.
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
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

    # Two hypotheses of one video must not replace each other's API inputs.
    result_key = hashlib.sha256(str(stats_path.resolve()).encode()).hexdigest()[:10]
    job_id = f"preloaded_{stem}_{result_key}"
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
    if request.args.get("contact_lab") == "1":
        return redirect(f"/contact-lab/{job_id}")
    return redirect(f"/results/{job_id}")


@app.route("/api/analysis_calibration", methods=["POST"])
def analysis_calibration():
    import cv2

    data = request.get_json(silent=True) or {}
    try:
        video_path = _resolve_video_path(str(data.get("video_path") or "").strip())
        seconds = float(data.get("seconds", 0))
        if not video_path.is_file() or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Choose an existing video and a nonnegative preview time")
        frame, last_second = _calibration_frame(video_path, seconds)
        height, width = frame.shape[:2]
        shared = PROJECT_ROOT / "court_calibration.json"
        source = json.loads(shared.read_text(encoding="utf-8")) if shared.is_file() else {}
        setup = _setup_store().get(data.get("setup_id")) if data.get("setup_id") else None
        if setup:
            source = setup.to_dict()
        sw, sh = source.get("frame_width", 3840), source.get("frame_height", 2160)
        corners = source.get("court_corners")
        net = source.get("net_line_points") or []
        left = source.get("net_x_left")
        right = source.get("net_x_right")
        top = source.get("net_y")
        if len(net) >= 2:
            (left, left_y), (right, right_y) = min(net), max(net)
        else:
            left_y = right_y = top
        if max(width, height) > 1600:
            factor = 1600 / max(width, height)
            frame = cv2.resize(frame, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("Could not encode preview frame")
        return jsonify(
            image="data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii"),
            frame_width=width, frame_height=height,
            last_second=last_second,
            court_corners=scale_corners(corners, (sw, sh), (width, height)) if corners else None,
            net_tape=[
                [width * (left / sw if left is not None else .15),
                 height * (left_y / sh if left_y is not None else .30)],
                [width * (right / sw if right is not None else .85),
                 height * (right_y / sh if right_y is not None else .30)],
            ],
            setups=[{"id": item.id, "name": item.name} for item in _setup_store().load()],
        )
    except cv2.error:
        app.logger.exception("Could not decode or encode calibration preview")
        return jsonify(error="Could not decode or encode the preview frame. Try another preview time."), 422
    except (ValueError, TypeError, OSError) as exc:
        return jsonify(error=str(exc)), 400


@app.route("/api/analysis_calibration/frame")
def analysis_calibration_frame():
    """Native-resolution preview frame for the calibration magnifier."""
    import cv2

    try:
        video_path = _resolve_video_path(request.args.get("video_path", "").strip())
        seconds = float(request.args.get("seconds", 0))
        if not video_path.is_file() or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Choose an existing video and a nonnegative preview time")
        frame, _ = _calibration_frame(video_path, seconds)
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise ValueError("Could not encode preview frame")
    except cv2.error:
        return jsonify(error="Could not decode the preview frame"), 422
    except (ValueError, TypeError, OSError) as exc:
        return jsonify(error=str(exc)), 400
    return Response(encoded.tobytes(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.route("/api/upload_video", methods=["GET", "POST"])
def upload_video():
    """Store a video dropped on the New analysis page (GET checks for a copy)."""
    try:
        if request.method == "GET":
            name = request.args.get("name", "")
            size = int(request.args.get("size", "-1"))
            path = find_existing_upload(UPLOAD_DIR, name, size)
            return jsonify(path=str(path) if path else None)
        name = urllib.parse.unquote(request.headers.get("X-File-Name", ""))
        size = request.content_length
        if size is None:
            raise UploadError("Upload needs a Content-Length header", status=411)
        path, reused = store_upload(UPLOAD_DIR, name, size, request.stream)
        return jsonify(path=str(path), reused=reused)
    except UploadError as exc:
        return jsonify(error=str(exc)), exc.status
    except (ValueError, OSError) as exc:
        return jsonify(error=str(exc)), 400


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    video_path_raw = str(data.get("video_path", "")).strip()
    skip_frames = int(data.get("skip_frames", 0))
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

    job_args = (str(video_path), skip_frames, refresh_net_floor, net_height_m)
    if "calibration" in data:
        try:
            edit = data["calibration"]
            frame, _ = _calibration_frame(video_path)
            height, width = frame.shape[:2]
            if (edit["frame_width"], edit["frame_height"]) != (width, height):
                raise ValueError("Video size changed; reload the calibration preview")
            calibration = _manual_calibration(edit, width, height, net_height_m)
            job_args = (str(video_path), skip_frames, False, net_height_m, calibration)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return jsonify(error=f"Invalid calibration: {exc}"), 400

    job_id = f"job_{time.time_ns()}"
    t = threading.Thread(
        target=_run_analysis,
        args=(job_id, *job_args),
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

    try:
        workspace_context = None
        if workspace_path(Path(job["stats_path"])).exists():
            with _player_contact_lock:
                state = read_workspace(job)
                stats = _prepare_viewer_stats(state["stats"])
                stats.setdefault("total_frames", state["total_frames"])
                workspace_context = score_projection(state)
        else:
            with open(job["stats_path"], encoding="utf-8") as f:
                stats = _prepare_viewer_stats(json.load(f))
        return render_template("viewer.html", job=job, stats=stats, workspace_context=workspace_context)
    except (StatisticsSourceChanged, OSError) as error:
        return jsonify(error=str(error)), 409
    except (ValueError, KeyError, TypeError) as error:
        return jsonify(error=str(error)), 400


@app.route("/api/annotations/<job_id>")
def annotations(job_id):
    job = _get_job(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Not ready"}), 404
    if not workspace_path(Path(job["stats_path"])).exists():
        return send_file(job["annotations_path"], mimetype="application/json")
    try:
        with _player_contact_lock:
            state = read_workspace(job, include_annotations=True)
            payload = state["annotations"]
            payload["player_contacts"] = state["report"]
            payload["workspace_revision"] = state["revision"]
            if state["context_saved"]:
                payload["match_context"] = state["config"]
                for row in payload.get("frames") or []:
                    segment, _ = context_at(state["config"], row["f"])
                    row["excluded_from_match_stats"] = segment.get("match_phase") == "non_game"
                    if segment.get("match_phase"):
                        row["match_phase"] = segment["match_phase"]
            return jsonify(payload)
    except (StatisticsSourceChanged, OSError) as error:
        return jsonify(error=str(error)), 409
    except (ValueError, KeyError, TypeError) as error:
        return jsonify(error=str(error)), 400


@app.route("/contact-lab/<job_id>")
def contact_lab_page(job_id):
    job = _get_job(job_id)
    if not job or job.get("status") != "done":
        return "Result not found", 404
    return render_template("contact_lab.html", job_id=job_id, job=job)


@app.route("/api/contact_lab/<job_id>")
def contact_lab_data(job_id):
    job = _get_job(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Result not found"}), 404
    path = Path(job["stats_path"])
    lab_path = path.with_name(path.name.removesuffix("_stats.json")+"_contact_lab.json")
    if not lab_path.is_file():
        return jsonify({"error": "This result has no volleyball contact experiment"}), 404
    try:
        stats_raw, ann_raw = path.read_bytes(), Path(job["annotations_path"]).read_bytes()
        return jsonify(load_lab(lab_path, source_fingerprint(stats_raw, ann_raw), json.loads(stats_raw)))
    except (ValueError, KeyError, TypeError, OSError) as error:
        return jsonify({"error": str(error)}), 409


_player_contact_lock = threading.Lock()


@app.route("/api/player_contacts/<job_id>", methods=["GET", "POST"])
def player_contacts(job_id):
    return _player_contact_response(job_id)


@app.route("/player-statistics/<job_id>")
def player_statistics(job_id):
    return _player_contact_response(job_id, statistics_page=True)


@app.post("/player-statistics/<job_id>/excel")
def player_statistics_excel(job_id):
    return _player_contact_response(job_id, statistics_page=True)


@app.route("/player-highlights/<job_id>")
def player_highlights_page(job_id):
    return _player_contact_response(job_id, statistics_page=True, highlights_page=True)


@app.route("/statistics-evidence/<job_id>")
def statistics_evidence_page(job_id):
    return _player_contact_response(job_id, statistics_page=True, evidence_page=True)


@app.route("/api/statistics/<job_id>", methods=["GET", "POST"])
def statistics_data(job_id):
    return _player_contact_response(job_id, statistics_page=True, statistics_api=True)


def _workspace_video_url(job):
    return (f"/api/original_video?path={job['original_video_path_url']}"
            if job.get("original_video_path_url") else url_for("video", job_id=job.get("job_id") or request.view_args["job_id"]))


@app.route("/match-setup/<job_id>")
def match_setup_page(job_id):
    return _match_workspace_response(job_id, "setup", page=True)


@app.route("/tactics/<job_id>")
def tactics_page(job_id):
    return _match_workspace_response(job_id, "tactics", page=True)


@app.route("/developer-review/<job_id>")
def developer_review_page(job_id):
    return _match_workspace_response(job_id, "developer", page=True)


@app.route("/api/match_context/<job_id>", methods=["GET", "POST"])
def match_context_data(job_id):
    return _match_workspace_response(job_id, "setup")


@app.route("/api/developer_review/<job_id>", methods=["GET", "POST"])
def developer_review_data(job_id):
    return _match_workspace_response(job_id, "developer")


@app.route("/api/tactics/<job_id>")
def tactics_data(job_id):
    return _match_workspace_response(job_id, "tactics")


def _match_workspace_response(job_id, tab, *, page=False):
    job = _get_job(job_id)
    if not job or job.get("status") != "done":
        return jsonify(error="Results not ready"), 404
    try:
        with _player_contact_lock:
            state = read_workspace(job, include_annotations=tab in {"developer", "tactics"})
            if request.method == "POST":
                body = request.get_json(silent=True)
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                if body.get("source") != state["source"] or body.get("revision") != state["revision"]:
                    raise StatisticsSourceChanged("Workspace changed; reload before saving")
                if tab == "setup":
                    payload = validate_context(body.get("config"), state["total_frames"], state["report"]["roster_ids"])
                    project_context(state["stats"], payload)
                    kind = "context"
                else:
                    from src.viewer_app.developer_review import validate_contact_edit, apply_contact_edits
                    payload = validate_contact_edit(body.get("edit"), state["stats"], state["report"], state["total_frames"])
                    projected, projected_report = apply_contact_edits(state["raw_stats"], state["legacy_report"], state["edits"] + [payload], author_proposals=state["raw_report"])
                    if state["context_saved"]:
                        projected = project_context(projected, state["config"])
                    build_player_match_statistics(projected, projected_report, state["point_reviews"])
                    kind = "contact"
                append_record(job, state["source"], state["revision"], kind, payload)
                state = read_workspace(job, include_annotations=tab in {"developer", "tactics"})
            metadata = {"source": state["source"], "revision": state["revision"], "config": state["config"],
                        "fps": state["fps"], "total_frames": state["total_frames"],
                        "roster_ids": state["report"]["roster_ids"], "history": state["history"],
                        "context_saved": state["context_saved"],
                        "context_source": "human_workspace" if state["context_saved"] else "saved_analysis_defaults"}
            api = {"setup": "match_context_data", "developer": "developer_review_data", "tactics": "tactics_data"}[tab]
            if page:
                template = {"setup": "match_setup.html", "developer": "developer_review.html", "tactics": "tactics.html"}[tab]
                return render_template(template, job=job, video_url=_workspace_video_url(job),
                    api_url=url_for(api, job_id=job_id), context_api_url=url_for("match_context_data", job_id=job_id),
                    point_api_url=url_for("statistics_data", job_id=job_id), active_tab=tab, **metadata)
            if tab == "developer":
                from src.viewer_app.developer_review import build_review_queue
                summary = build_player_match_statistics(state["stats"], state["report"], state["point_reviews"])
                queue = build_review_queue(state["stats"], state["report"], summary["reconciliation"], state["annotations"])
                return jsonify(dict(queue=queue, report=state["report"], reconciliation=summary["reconciliation"], **metadata))
            if tab == "tactics":
                from src.viewer_app.tactics import build_tactics
                tactics = build_tactics(state["stats"], state["report"], state["annotations"], state["config"], state["point_reviews"])
                return jsonify(dict(tactics, **metadata))
            return jsonify(metadata)
    except (StatisticsSourceChanged, ExperimentSourceChanged) as error:
        return jsonify(error=str(error)), 409
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        return jsonify(error=str(error)), 400
    except OSError as error:
        return jsonify(error=str(error)), 409


def _player_contact_response(job_id, *, statistics_page=False, highlights_page=False,
                             evidence_page=False, statistics_api=False):
    job = _get_job(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "Results not ready"}), 404
    try:
        with _player_contact_lock:
            body = request.get_json(silent=True) if request.is_json else {}
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object")
            method = None if statistics_page else body.get("method") or request.args.get("method")
            state = read_workspace(job, method=method)
            source, report = state["source"], state["report"]
            stats_path = Path(job["stats_path"])
            if request.method == "POST" and not statistics_page:
                if body.get("source") != source or ((state["history"] or body.get("revision")) and body.get("revision") != state["revision"]):
                    raise StatisticsSourceChanged("Analysis or workspace changed; reload contacts before reviewing")
                if type(body.get("event_index")) is not int:
                    raise ValueError("event_index must be an integer")
                if type(body.get("other_side", False)) is not bool or type(body.get("reset", False)) is not bool:
                    raise ValueError("other_side and reset must be booleans")
                if body.get("player_id") is not None and not isinstance(body["player_id"], str):
                    raise ValueError("player_id must be a roster ID or null")
                review = {key: body[key] for key in ("event_index", "player_id", "other_side", "reset") if key in body}
                if state["history"]:
                    from src.viewer_app.developer_review import validate_contact_edit, apply_contact_edits
                    edit = ({"action": "reset", "event_index": body["event_index"], "axes": ["author"]}
                            if body.get("reset") else {"action": "update", "event_index": body["event_index"],
                            "changes": {"player_id": body.get("player_id"), "other_side": body.get("other_side", False)}})
                    edit = validate_contact_edit(edit, state["stats"], report, state["total_frames"])
                    apply_contact_edits(state["raw_stats"], state["legacy_report"], state["edits"] + [edit], author_proposals=state["raw_report"])
                    append_record(job, source, state["revision"], "contact", edit)
                else:
                    apply_reviews(state["raw_report"], state["author_reviews"] + [review])
                    guard_source(job, source)
                    record = {"schema": "player_contact_review_v1", "source": source,
                              "saved_at": datetime.now(timezone.utc).isoformat(), "review": review}
                    with author_review_path(stats_path).open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                state = read_workspace(job, method=method)
                report = state["report"]
            stats = state["stats"]
            if highlights_page:
                player = request.args.get("player") or (report["roster_ids"] or [None])[0]
                if player not in report["roster_ids"]:
                    return "Unknown roster player", 404
                moments = player_highlights(report, stats.get("all_events") or [], stats["fps"], player,
                                           include_suggestions=request.args.get("suggestions") == "1")
                return render_template("player_highlights.html", job=job, player=player, moments=moments,
                    roster=report["roster_ids"], categories=HIGHLIGHT_CATEGORIES, fps=float(stats["fps"]),
                    video_url=_workspace_video_url(job), include_suggestions=request.args.get("suggestions") == "1",
                    position_labels=PLAYER_POSITIONS, portraits={p: len(c) for p, c in state["references"].items()})
            if statistics_page:
                if statistics_api and request.method == "POST":
                    if body.get("source") != source or ((state["history"] or body.get("revision")) and body.get("revision") != state["revision"]):
                        raise StatisticsSourceChanged("Analysis or workspace changed; reload statistics before reviewing")
                    review = validate_point_review(body)
                    current = build_player_match_statistics(stats, report, state["point_reviews"])
                    if review["attempt_ref"] not in {p["attempt_ref"] for p in current["reconciliation"]["points"]}:
                        raise ValueError("Point review refers to an unknown attempt")
                    build_player_match_statistics(stats, report, state["point_reviews"] + [review])
                    guard_source(job, source)
                    append_point_review(stats_path, source, review)
                    state = read_workspace(job)
                    stats, report = state["stats"], state["report"]
                summary = build_player_match_statistics(stats, report, state["point_reviews"])
                if statistics_api:
                    return jsonify(dict(summary, source=source, revision=state["revision"],
                                        point_review_count=len(state["point_reviews"])))
                if evidence_page:
                    player = request.args.get("player") or "all"
                    kind, metric = request.args.get("kind") or "attack", request.args.get("metric") or "total"
                    evidence = statistic_evidence(summary, player, kind, metric)
                    return render_template("statistics_evidence.html", job=job, summary=summary,
                        source=source, revision=state["revision"], evidence=evidence, player=player, kind=kind,
                        metric=metric, modes=STATISTICS_VIEWS, fps=float(stats["fps"]), video_url=_workspace_video_url(job))
                if request.endpoint == "player_statistics_excel":
                    submitted = json.loads(request.form.get("profiles", "{}"))
                    if not isinstance(submitted, dict):
                        raise ValueError("Expected player profiles")
                    profiles = {}
                    for player in summary["players"]:
                        profile = submitted.get(player, {})
                        if not isinstance(profile, dict):
                            raise ValueError("Invalid player profile")
                        name, number, position = (profile.get(key, "") for key in ("name", "number", "position"))
                        if not all(isinstance(value, str) for value in (name, number, position)):
                            raise ValueError("Player name, jersey and position must be text")
                        if position and position not in PLAYER_POSITIONS:
                            raise ValueError("Unknown player position")
                        profiles[player] = {"name": name.strip()[:64], "number": number.strip()[:8], "position": position}
                    return send_file(export_player_statistics(summary, profiles), as_attachment=True,
                        download_name=f"{job['video_stem']}-player-statistics.xlsx",
                        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                selected = request.args.get("player")
                return render_template("player_statistics.html", job=job, summary=summary,
                    selected_player=selected if selected in summary["players"] else None,
                    contact_counts=summary["touch_counts"], position_labels=PLAYER_POSITIONS,
                    modes=STATISTICS_VIEWS, stat_value=player_statistic_value,
                    portraits={player: len(choices) for player, choices in state["references"].items()},
                    portrait_version=source["annotations_sha256"])
            report.update(source=source, revision=state["revision"], review_count=len(state["author_reviews"]),
                          method_id=state["method_id"], methods=state["bundle"]["methods"] if state["bundle"] else {},
                          experiment_selection=state["bundle"].get("selection") if state["bundle"] else None)
            return jsonify(report)
    except (ExperimentSourceChanged, StatisticsSourceChanged) as error:
        return jsonify({"error": str(error)}), 409
    except (ValueError, KeyError, TypeError) as error:
        return jsonify({"error": str(error)}), 400
    except OSError as error:
        return jsonify({"error": str(error)}), 409


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


def _recompute_attack_heatmap(job: dict, payload: dict, *, events=None) -> dict:
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

    if events is None:
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
        **{key: payload.get(key) for key in (
            "net_tape_top", "net_tape_bottom", "net_tape_endpoint_y",
            "net_floor_left", "net_floor_right",
        )},
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
        previous_net_y = payload.get("net_y")
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
            payload[key] = number

        _sync_manual_net_geometry(payload, previous_net_y)

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
    stats_path = Path(job["stats_path"])
    if any(path.exists() for path in (workspace_path(stats_path), author_review_path(stats_path),
                                       stats_path.with_name(stats_path.name.removesuffix("_stats.json") + "_point_reviews.jsonl"))):
        try:
            with _player_contact_lock:
                state = read_workspace(job, include_annotations=True)
                events = [event for event in state["stats"].get("all_events") or []
                          if not event.get("review_excluded") and not event.get("excluded_from_match_stats")]
                heatmap = _recompute_attack_heatmap(job, state["annotations"], events=events)
                return jsonify(attack_heatmap=heatmap, persisted=False, source="reviewed_workspace")
        except (StatisticsSourceChanged, OSError) as error:
            return jsonify(error=str(error)), 409
        except (CourtSetupError, ValueError, KeyError, TypeError) as error:
            return jsonify(error=str(error)), 400
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
_THUMB_CACHE_SIZE = 256
_thumb_cache: "OrderedDict[tuple, bytes]" = OrderedDict()
_thumb_lock = threading.Lock()


@app.route("/api/player-portrait/<job_id>/<player_id>")
def player_portrait(job_id, player_id):
    job = _get_job(job_id)
    if not job or job["status"] != "done":
        return "Result not found", 404
    try:
        with _player_contact_lock:
            report, _, _, references, _ = _load_player_contact_inputs(job)
        choices = references.get(player_id, [])
        index = request.args.get("candidate", default=0, type=int)
        original = job.get("original_video_path")
        source = _resolve_video_path(original) if original else None
        if (player_id not in report["roster_ids"] or not 0 <= index < len(choices)
                or not source or not source.is_file()):
            return "Portrait unavailable", 404
        info = source.stat()
        data = _decode_player_portrait(source, info.st_mtime_ns, info.st_size, choices[index])
    except (ValueError, KeyError, TypeError, OSError, ExperimentSourceChanged):
        return "Portrait unavailable", 404
    if data is None:
        return "Portrait unavailable", 404
    return Response(data, mimetype="image/jpeg")


@app.route("/api/frame/<job_id>/<int:frame>")
def frame_thumbnail(job_id, frame):
    """One downscaled JPEG of a source frame, e.g. a rally's first frame."""
    job = _get_job(job_id)
    if not job or job["status"] != "done":
        return ("Not ready", 404)
    # The original video is what the overlays are drawn on; an analysed MP4
    # is only a fallback for results whose original has moved.
    source = next(
        (Path(path) for path in (job.get("original_video_path"), job.get("video_path"))
         if path and Path(path).is_file()),
        None,
    )
    if source is None:
        return ("Video not found", 404)
    key = (str(source), frame)
    with _thumb_lock:
        data = _thumb_cache.get(key)
    if data is None:
        data = _decode_thumbnail(source, frame)
        if data is None:
            return ("Frame not readable", 404)
        with _thumb_lock:
            _thumb_cache[key] = data
            while len(_thumb_cache) > _THUMB_CACHE_SIZE:
                _thumb_cache.popitem(last=False)
    return Response(data, mimetype="image/jpeg",
                    headers={"Cache-Control": "private, max-age=86400"})


if __name__ == "__main__":
    # A second instance is useful while another viewer is already serving 5050.
    port = int(os.environ.get("VIEWER_PORT", 5050))
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    print("\nVolleyball Analytics Viewer")
    print(f"Open http://localhost:{port} in your browser")
    print(f"Processed results: http://localhost:{port}/library\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
