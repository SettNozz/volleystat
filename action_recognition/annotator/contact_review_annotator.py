"""Review the spotter's contacts: type, side, false positives, misses.

Everything the dense contact spotter produces is a model prediction. This turns
a batch of analyses into a review queue where each contact is confirmed,
retyped, given a side, or thrown out, and where a contact the model missed can
be inserted. Nothing in it is ground truth until a human has touched it:

* every record starts ``label_source: "model"`` and ``verified: false``;
* a record becomes ``label_source: "human"``, ``verified: true`` only through a
  keystroke here, and keeps the model's original prediction beside the
  correction so the two never blur;
* the review file lives in its own directory and never writes into
  ``data/full_video_gold`` or any other authoritative set.

    python -m action_recognition.annotator.contact_review_annotator --help
    # http://localhost:5059

Keys: 1-6 set the type, Enter keeps the predicted one, A/B the side, X rejects,
S marks unsure, Z steps back, + inserts a missed contact in the gap ahead.
"""

from __future__ import annotations

import argparse
import io
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file

from action_recognition.rally_spotting import dense_contacts as dc
from action_recognition.rally_spotting import team_attribution as ta

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "contact_review_v1"
RENDER_WIDTH = 1600
CONTACT_TYPES = list(dc.CONTACT_TYPES)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return dc.file_sha256(path)


def build_queue(annotations_dir: Path) -> dict[str, Any]:
    """One review record per spotted contact, across every analysis found."""
    annotations_dir = Path(annotations_dir)
    files = sorted(annotations_dir.glob("*_annotations.json"))
    if not files:
        raise FileNotFoundError(f"no *_annotations.json under {annotations_dir}")

    videos: dict[str, Any] = {}
    contacts: list[dict[str, Any]] = []
    for path in files:
        stem = path.name[: -len("_annotations.json")]
        payload = json.loads(path.read_text(encoding="utf-8"))
        spotter = payload.get("contact_spotter") or {}
        if not spotter.get("enabled"):
            raise ValueError(
                f"{path.name} carries no contact_spotter block; rerun the "
                "analysis with --contact-spotter"
            )
        ball = {
            int(frame["f"]): frame.get("ball")
            for frame in payload["frames"]
        }
        videos[stem] = {
            "stem": stem,
            "video_path": payload.get("original_video_path"),
            "fps": payload.get("fps"),
            "frame_width": payload.get("frame_width"),
            "frame_height": payload.get("frame_height"),
            "num_frames": len(payload["frames"]),
            "net_y": payload.get("net_y"),
            "net_floor_left": payload.get("net_floor_left"),
            "net_floor_right": payload.get("net_floor_right"),
            "annotations": str(path.resolve()),
            "annotations_sha256": _sha256(path),
            "spotter_checkpoint": spotter.get("checkpoint"),
            "spotter_diagnostics": spotter.get("diagnostics"),
            "ball_by_frame": ball,
        }
        # A candidates file, where one exists, is what the review queue is
        # built from: it is generated for recall rather than for the most
        # likely sequence, and a reviewer is better served by rejecting a
        # spare card than by hunting for a contact nobody offered. Without
        # one, fall back to the decoder's own events in the stats file.
        candidates_path = path.with_name(f"{stem}_candidates.json")
        stats_path = path.with_name(f"{stem}_stats.json")
        if candidates_path.exists():
            payload_candidates = json.loads(
                candidates_path.read_text(encoding="utf-8")
            )
            videos[stem]["candidates"] = str(candidates_path.resolve())
            videos[stem]["candidates_sha256"] = _sha256(candidates_path)
            videos[stem]["candidate_stats"] = payload_candidates.get("stats")
            videos[stem]["candidate_threshold"] = payload_candidates.get("threshold")
            events = sorted(
                payload_candidates.get("candidates") or [],
                key=lambda row: int(row.get("frame", 0)),
            )
        else:
            if not stats_path.exists():
                raise FileNotFoundError(
                    f"{stats_path.name} is missing; the spotted events live there"
                )
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
            events = sorted(
                (
                    event for event in stats.get("all_events") or []
                    if event.get("source") == "dense_contact_spotter"
                ),
                key=lambda row: int(row.get("frame", 0)),
            )
        if stats_path.exists():
            videos[stem]["stats"] = str(stats_path.resolve())
            videos[stem]["stats_sha256"] = _sha256(stats_path)
        for event in events:
            frame = int(event["frame"])
            contacts.append({
                "id": f"{stem}:{frame}",
                "video_id": stem,
                "frame": frame,
                "predicted_type": event.get("type"),
                "predicted_core": event.get("core_type"),
                "confidence": event.get("confidence"),
                # Whether the constrained decoder also chose this one. A
                # candidate it rejected is likelier to be spare, and the page
                # says so rather than presenting all cards as equal.
                "decoded": event.get("decoded"),
                "peak_type": event.get("peak_type"),
                "origin": "model",
                "label_source": "model",
                "verified": False,
                "review": None,
            })
    contacts.sort(key=lambda row: (row["video_id"], row["frame"]))
    return {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "annotations_dir": str(annotations_dir.resolve()),
        "provenance": {
            "note": (
                "Model output under review. Nothing here is ground truth "
                "until verified is true and label_source is human."
            ),
            "videos": {
                stem: {
                    key: video[key] for key in (
                        "annotations", "annotations_sha256", "stats",
                        "stats_sha256", "candidates", "candidates_sha256",
                        "candidate_stats", "candidate_threshold",
                        "spotter_checkpoint", "spotter_diagnostics",
                    ) if key in video
                }
                for stem, video in videos.items()
            },
        },
        "videos": {
            stem: {k: v for k, v in video.items() if k != "ball_by_frame"}
            for stem, video in videos.items()
        },
        "contacts": contacts,
        "history": [],
        "_ball": {stem: video["ball_by_frame"] for stem, video in videos.items()},
    }


def create_app(annotations_dir: Path, output: Path) -> Flask:
    output = Path(output).resolve()
    fresh = build_queue(Path(annotations_dir))
    ball_by_video = fresh.pop("_ball")
    if output.exists():
        project = json.loads(output.read_text(encoding="utf-8"))
        def fingerprint(videos: dict) -> dict:
            return {
                stem: (
                    video.get("annotations_sha256"),
                    video.get("stats_sha256"),
                    video.get("candidates_sha256"),
                )
                for stem, video in videos.items()
            }

        saved = fingerprint(project.get("videos") or {})
        current = fingerprint(fresh["videos"])
        if saved != current:
            raise ValueError(
                "the analyses changed since this review was started; use a "
                "new output path rather than mixing two runs"
            )
    else:
        project = fresh

    app = Flask(__name__)
    lock = threading.Lock()
    decode_lock = threading.Lock()
    captures: dict[str, Any] = {}

    def index_of(identity: str) -> int:
        for position, contact in enumerate(project["contacts"]):
            if contact["id"] == identity:
                return position
        return -1

    def persist() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(".tmp")
        temp.write_text(
            json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        temp.replace(output)

    def counts() -> dict[str, int]:
        reviewed = [c for c in project["contacts"] if c["verified"]]
        return {
            "total": len(project["contacts"]),
            "verified": len(reviewed),
            "rejected": sum(
                1 for c in reviewed if c["review"]["status"] == "rejected"
            ),
            "added": sum(1 for c in project["contacts"] if c["origin"] == "human"),
            "retyped": sum(
                1 for c in reviewed
                if c["review"]["status"] == "kept"
                and c["review"]["type"] != c["predicted_type"]
            ),
        }

    @app.after_request
    def cache_policy(response):
        if request.path.startswith("/api/frame"):
            response.headers["Cache-Control"] = "private, max-age=3600"
        else:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return Path(__file__).with_name("contact_review.html").read_text(
            encoding="utf-8",
        )

    @app.get("/api/project")
    def get_project():
        with lock:
            return jsonify({
                "schema": project["schema"],
                "videos": project["videos"],
                "contacts": project["contacts"],
                "contact_types": CONTACT_TYPES,
                "counts": counts(),
            })

    def _render(image: np.ndarray, stem: str, frame_number: int) -> np.ndarray:
        video = project["videos"][stem]
        height, width = image.shape[:2]
        scale = RENDER_WIDTH / float(width)
        canvas = cv2.resize(image, (RENDER_WIDTH, int(round(height * scale))))
        left = video.get("net_floor_left")
        right = video.get("net_floor_right")
        if left and right:
            start = (int(round(left[0] * scale)), int(round(left[1] * scale)))
            end = (int(round(right[0] * scale)), int(round(right[1] * scale)))
            if end[0] != start[0]:
                slope = (end[1] - start[1]) / (end[0] - start[0])
                start = (0, int(round(start[1] - slope * start[0])))
                end = (canvas.shape[1],
                       int(round(end[1] + slope * (canvas.shape[1] - end[0]))))
            cv2.line(canvas, start, end, (255, 0, 255), 3)
            mid = (start[1] + end[1]) // 2
            cv2.putText(canvas, "A far", (20, max(34, mid - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
            cv2.putText(canvas, "B near", (20, min(canvas.shape[0] - 14, mid + 48)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
        ball = ball_by_video.get(stem, {}).get(frame_number)
        if ball:
            centre = (int(round(ball[0] * scale)), int(round(ball[1] * scale)))
            cv2.circle(canvas, centre, 24, (0, 255, 255), 3)
            for dx, dy in ((-40, 0), (40, 0), (0, -40), (0, 40)):
                cv2.line(canvas, (centre[0] + dx // 2, centre[1] + dy // 2),
                         (centre[0] + dx, centre[1] + dy), (0, 255, 255), 3)
        return canvas

    @app.get("/api/ball/<stem>/<int:frame_number>")
    def get_ball(stem: str, frame_number: int):
        """Where the detector put the ball on a frame, if anywhere."""
        if stem not in project["videos"]:
            return jsonify(error="Unknown video"), 404
        return jsonify({"ball": ball_by_video.get(stem, {}).get(frame_number)})

    @app.get("/api/frame/<stem>/<int:frame_number>")
    def get_frame(stem: str, frame_number: int):
        video = project["videos"].get(stem)
        if video is None:
            return jsonify(error="Unknown video"), 404
        if not 0 <= frame_number < int(video["num_frames"]):
            return jsonify(error="Frame outside video"), 400
        with decode_lock:
            capture = captures.get(stem)
            if capture is None:
                capture = captures[stem] = cv2.VideoCapture(video["video_path"])
            if int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) != frame_number:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, image = capture.read()
            actual = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            if not ok or actual != frame_number:
                return jsonify(error="Native frame decode failed"), 500
        canvas = _render(image, stem, frame_number)
        ok, encoded = cv2.imencode(
            ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 85],
        )
        if not ok:
            return jsonify(error="Image encoding failed"), 500
        return send_file(io.BytesIO(encoded.tobytes()), mimetype="image/jpeg")

    @app.post("/api/review")
    def save_review():
        if request.headers.get("Sec-Fetch-Site", "same-origin") != "same-origin":
            return jsonify(error="Cross-site write refused"), 403
        payload = request.get_json(silent=True) or {}
        identity = payload.get("id")
        status = payload.get("status")
        position = index_of(identity)
        if position < 0:
            return jsonify(error="Unknown contact"), 400
        if status not in ("kept", "rejected", "unsure", None):
            return jsonify(error="Invalid status"), 400
        contact_type = payload.get("type")
        team = payload.get("team")
        if status == "kept":
            if contact_type not in CONTACT_TYPES:
                return jsonify(error="Invalid contact type"), 400
            if team not in ("team_a", "team_b", "unknown"):
                return jsonify(error="Invalid team"), 400

        # The reviewer may land on a different frame than the model proposed,
        # and may place the ball where the detector lost it. Both are stored
        # beside the model's own answer, never over it.
        contact = project["contacts"][position]
        video = project["videos"][contact["video_id"]]
        frame_override = payload.get("frame")
        if frame_override is not None:
            if (isinstance(frame_override, bool)
                    or not isinstance(frame_override, int)
                    or not 0 <= frame_override < int(video["num_frames"])):
                return jsonify(error="Frame outside video"), 400
        ball_override = payload.get("ball")
        if ball_override is not None:
            if (not isinstance(ball_override, (list, tuple))
                    or len(ball_override) != 2):
                return jsonify(error="Ball must be a pair of coordinates"), 400
            try:
                ball_override = [float(ball_override[0]), float(ball_override[1])]
            except (TypeError, ValueError):
                return jsonify(error="Ball coordinates must be numbers"), 400
            if not (0 <= ball_override[0] < float(video["frame_width"])
                    and 0 <= ball_override[1] < float(video["frame_height"])):
                return jsonify(error="Ball outside the frame"), 400
        with lock:
            if status is None:
                contact["review"] = None
                contact["verified"] = False
                contact["label_source"] = contact["origin"]
            else:
                contact["review"] = {
                    "status": status,
                    "type": contact_type if status == "kept" else None,
                    "core_type": (
                        dc.to_core(contact_type) if status == "kept" else None
                    ),
                    "team": team if status == "kept" else None,
                    # None means "the model's frame was right"; a number is a
                    # correction, which the gold audit needed for 155 of 410.
                    "frame": frame_override,
                    "ball": ball_override,
                    "reviewed_at": _utc_now(),
                }
                contact["verified"] = status != "unsure"
                contact["label_source"] = (
                    "human" if status != "unsure" else contact["origin"]
                )
            project["history"].append({
                "id": identity, "status": status, "at": _utc_now(),
            })
            persist()
            return jsonify({"ok": True, "counts": counts(), "contact": contact})

    @app.post("/api/insert")
    def insert_contact():
        if request.headers.get("Sec-Fetch-Site", "same-origin") != "same-origin":
            return jsonify(error="Cross-site write refused"), 403
        payload = request.get_json(silent=True) or {}
        stem = payload.get("video_id")
        frame_number = payload.get("frame")
        contact_type = payload.get("type")
        team = payload.get("team")
        video = project["videos"].get(stem)
        if video is None:
            return jsonify(error="Unknown video"), 400
        if (isinstance(frame_number, bool) or not isinstance(frame_number, int)
                or not 0 <= frame_number < int(video["num_frames"])):
            return jsonify(error="Frame outside video"), 400
        if contact_type not in CONTACT_TYPES:
            return jsonify(error="Invalid contact type"), 400
        if team not in ("team_a", "team_b", "unknown"):
            return jsonify(error="Invalid team"), 400
        identity = f"{stem}:{frame_number}"
        with lock:
            if index_of(identity) >= 0:
                return jsonify(error="A contact already exists on that frame"), 409
            contact = {
                "id": identity,
                "video_id": stem,
                "frame": frame_number,
                # A human-added contact has no prediction to compare against,
                # and must never look like one the model found.
                "predicted_type": None,
                "predicted_core": None,
                "confidence": None,
                "origin": "human",
                "label_source": "human",
                "verified": True,
                "review": {
                    "status": "added",
                    "type": contact_type,
                    "core_type": dc.to_core(contact_type),
                    "team": team,
                    "reviewed_at": _utc_now(),
                },
            }
            project["contacts"].append(contact)
            project["contacts"].sort(key=lambda row: (row["video_id"], row["frame"]))
            project["history"].append({
                "id": identity, "status": "added", "at": _utc_now(),
            })
            persist()
            return jsonify({
                "ok": True, "counts": counts(),
                "index": index_of(identity),
                "contacts": project["contacts"],
            })

    @app.get("/api/summary")
    def summary():
        with lock:
            rows = counts()
            per_video: dict[str, dict[str, int]] = {}
            for contact in project["contacts"]:
                bucket = per_video.setdefault(
                    contact["video_id"],
                    {"total": 0, "verified": 0, "rejected": 0, "added": 0},
                )
                bucket["total"] += 1
                if contact["verified"]:
                    bucket["verified"] += 1
                    if contact["review"]["status"] == "rejected":
                        bucket["rejected"] += 1
                if contact["origin"] == "human":
                    bucket["added"] += 1
            agreement = {"type_kept": 0, "type_changed": 0}
            for contact in project["contacts"]:
                review = contact["review"]
                if not review or review["status"] != "kept":
                    continue
                if review["type"] == contact["predicted_type"]:
                    agreement["type_kept"] += 1
                else:
                    agreement["type_changed"] += 1
            return jsonify({
                "counts": rows, "per_video": per_video, "type": agreement,
            })

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir", type=Path,
        default=ROOT / "results/spotter_night_20261029_v1",
        help="directory of finished analyses run with --contact-spotter",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/contact_review_20261029_v1/review.json",
    )
    parser.add_argument("--port", type=int, default=5059)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)

    app = create_app(args.annotations_dir, args.output)
    print(f"[contact-review] review -> {args.output}")
    print(f"[contact-review] http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
