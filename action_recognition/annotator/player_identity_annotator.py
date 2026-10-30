"""Local human seed and independent checks for the player identity pilot."""
from __future__ import annotations

import argparse
import io
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, send_file

from scripts.evaluation.run_player_identity_pilot import digest, rerun
from src.tracking.player_identity import SCHEMA, default_seed, score


def create_app(project_dir: Path, comparison_path: Path | None = None) -> Flask:
    project_dir = project_dir.resolve()
    observations_path = project_dir/"observations.json"
    observations = json.loads(observations_path.read_text(encoding="utf-8"))
    source = observations["source"]
    if digest(Path(source["annotations"])) != source["annotations_sha256"]:
        raise ValueError("Source annotations changed; build a new versioned project")
    video = Path(source["video"])
    if video.stat().st_size != source["video_size"] or video.stat().st_mtime_ns != source["video_mtime_ns"]:
        raise ValueError("Source video changed; build a new versioned project")
    fingerprint = digest(observations_path)
    labels_path = project_dir/"labels.json"
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        if labels.get("observations_sha256") != fingerprint:
            raise ValueError("Observations changed; refusing to reuse human labels")
    else:
        labels = {"schema": SCHEMA, "observations_sha256": fingerprint, "seed": None, "checks": []}
    frames = {f["f"]: f for f in observations["frames"]}
    extra = None
    if comparison_path is not None:
        extra = json.loads(Path(comparison_path).read_text(encoding="utf-8"))
        if extra.get("observations_sha256") != fingerprint:
            raise ValueError("Extra comparison belongs to different observations")
        if not labels.get("seed") or extra["seed"]["assignments"] != labels["seed"]["assignments"] or extra["seed"]["frame"] != labels["seed"]["frame"]:
            raise ValueError("Extra comparison seed differs from human roster")
        expected = {p["key"] for f in observations["frames"] for p in f["players"]}
        for method in extra["methods"].values():
            if set(method["predictions"]) != expected:
                raise ValueError("Extra comparison detection keys differ from original cache")
    lock = threading.RLock()
    app = Flask(__name__)
    app.config["IDENTITY_PROJECT"] = project_dir

    def latest():
        path = sorted(project_dir.glob("comparison_*.json"))[-1]
        result = json.loads(path.read_text(encoding="utf-8"))
        if extra is not None:
            # Frozen experimental predictions; only rescore new human checks.
            for name, method in extra["methods"].items():
                if name in result["methods"]:
                    continue
                method = dict(method)
                method["human_checks"] = score(method["predictions"], labels["checks"], set(labels["seed"]["assignments"]))
                result["methods"][name] = method
        return result

    @app.get("/")
    def index():
        return send_file(Path(__file__).with_name("player_identity.html"))

    @app.get("/api/project")
    def state():
        with lock:
            result = latest()
            return jsonify({"fps": observations["fps"], "side": observations["side"],
                            "width": observations["width"], "height": observations["height"],
                            "roster_size": observations["roster_size"],
                            "frames": [{"f": f["f"], "players": [{k: v for k, v in p.items() if k != "appearance"}
                                        for p in f["players"]]} for f in observations["frames"]],
                            "labels": labels, "initial_seed": default_seed(observations["frames"], observations["roster_size"]),
                            "comparison": result})

    @app.get("/frame/<int:frame>")
    def image(frame: int):
        if frame not in frames:
            return jsonify(error="Frame not in cache"), 404
        with lock:
            cap = cv2.VideoCapture(str(video))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, image = cap.read()
            finally:
                cap.release()
            if not ok:
                return jsonify(error="Cannot decode frame"), 422
            width = min(1600, image.shape[1])
            image = cv2.resize(image, (width, round(width*image.shape[0]/image.shape[1])))
            ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
            if not ok:
                return jsonify(error="Cannot encode frame"), 422
            return send_file(io.BytesIO(buffer.tobytes()), mimetype="image/jpeg")

    @app.get("/video")
    def original_video():
        return send_file(video, conditional=True)

    @app.post("/api/labels")
    def save():
        payload = request.get_json(silent=True) or {}
        frame = payload.get("frame")
        role, assignments = payload.get("role"), payload.get("assignments")
        if not isinstance(frame, int) or frame not in frames or role not in ("seed", "check") or not isinstance(assignments, dict):
            return jsonify(error="Invalid frame, role or assignments"), 400
        available = {p["key"] for p in frames[frame]["players"]}
        allowed = {f"P{i+1}" for i in range(observations["roster_size"])}
        values = [v for v in assignments.values() if v is not None]
        if not set(assignments) <= available or any(not isinstance(v, str) or v not in allowed for v in values):
            return jsonify(error="Unknown detection/player ID"), 400
        if len(values) != len(set(values)):
            return jsonify(error="One player ID cannot label two detections in one frame"), 400
        if role == "seed" and (set(assignments) != available or set(values) != allowed or len(values) != len(available)):
            return jsonify(error="Seed requires all six distinct roster IDs in a complete frame"), 400
        with lock:
            timestamp = datetime.now(timezone.utc).isoformat()
            provenance = {"label_source": "human", "verified": True, "updated_at": timestamp}
            if role == "seed":
                # Once checks exist, roster meaning must stay stable for honest scores.
                if labels["checks"] and (labels["seed"]["frame"] != frame or labels["seed"]["assignments"] != assignments):
                    return jsonify(error="Checks already exist. Use a new project to redefine initial IDs."), 409
                labels["seed"] = {"frame": frame, "assignments": assignments, **provenance}
            else:
                if labels["seed"] is None:
                    return jsonify(error="Confirm initial IDs first"), 400
                if frame == labels["seed"]["frame"]:
                    return jsonify(error="Seed frame cannot also be a test frame"), 400
                labels["checks"] = [c for c in labels["checks"] if int(c["frame"]) != frame]
                labels["checks"].extend({"key": key, "frame": frame, "player_id": value, **provenance}
                                         for key, value in assignments.items())
            temporary = labels_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(labels_path)
            return jsonify(labels)

    @app.post("/api/compare")
    def run():
        with lock:
            rerun(project_dir)
            return jsonify(latest())

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5063)
    parser.add_argument("--extra-comparison", type=Path, help="Frozen experimental predictions on the same observations/seed")
    args = parser.parse_args()
    create_app(args.project, args.extra_comparison).run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
