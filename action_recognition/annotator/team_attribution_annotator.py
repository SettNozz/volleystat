"""Label which side made each reviewed contact, one keystroke per contact.

The contact spotter never says who touched the ball, and match statistics are
close to useless without it. Before annotating 400 contacts by hand it is worth
knowing whether geometry already answers the question -- so this records a
human label and a geometric prediction for the same contact, and reports how
often they agree.

**The prediction is hidden until the label is given.** Showing it first would
anchor the answer and the agreement number would then measure nothing.

    python -m action_recognition.annotator.team_attribution_annotator --help
    # http://localhost:5057

``team_a`` is the far side of the court, above the divider in frame;
``team_b`` is the near side. The page draws the divider and labels both halves,
so the choice is never ambiguous.

Keys: A far side, B near side, S unsure, Z back, arrows to move.
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
SCHEMA = "contact_team_labels_v1"
RENDER_WIDTH = 1600


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_project(
    reviews_path: Path, annotations_dir: Path,
) -> dict[str, Any]:
    """Every reviewable contact with its frame, ball pixel and geometry."""
    review = dc.reviewed_contacts(reviews_path)
    raw = json.loads(Path(reviews_path).read_text(encoding="utf-8"))
    events = {event["id"]: event for event in raw["events"]}

    analyses: dict[str, dict[str, Any]] = {}
    poses_by_frame: dict[str, dict[int, list]] = {}
    geometry: dict[str, dict[str, Any]] = {}
    for stem in sorted(raw["videos"]):
        path = Path(annotations_dir) / f"{stem}_annotations.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; this needs a pose-enabled analysis of "
                "every reviewed video"
            )
        annotations = json.loads(path.read_text(encoding="utf-8"))
        analyses[stem] = annotations
        poses_by_frame[stem] = {
            int(frame["f"]): (frame.get("poses") or [])
            for frame in annotations["frames"]
        }
        geometry[stem] = {
            "net_floor_left": annotations.get("net_floor_left"),
            "net_floor_right": annotations.get("net_floor_right"),
            "net_y": annotations.get("net_y"),
        }

    items: list[dict[str, Any]] = []
    for identity, event in sorted(events.items()):
        answer = raw["reviews"][identity]
        if answer.get("status") == "wrong_event":
            continue
        stem = event["video_id"]
        frame = int(answer["frame"])
        ball = answer.get("xy") if answer.get("spatial_valid") else None
        attribution = ta.attribute(
            poses_by_frame[stem].get(frame, []),
            ball,
            geometry[stem]["net_floor_left"],
            geometry[stem]["net_floor_right"],
        )
        items.append({
            "id": identity,
            "video_id": stem,
            "frame": frame,
            "contact_type": answer.get("label"),
            "raw_labels": event.get("raw_labels") or [],
            "ball": ball,
            "ball_status": answer.get("status"),
            "attribution": attribution,
        })

    # Chronological, not by event id. A string sort puts frame 10118 before
    # 903, which makes every frame request a backwards seek in a 4K file and
    # costs seconds each; it also scatters the match, and the labeller reads
    # the action better in order.
    items.sort(key=lambda item: (item["video_id"], item["frame"]))

    return {
        "schema": SCHEMA,
        "created_at": _utc_now(),
        "reviews_path": str(Path(reviews_path).resolve()),
        "review_sha256": review["review_sha256"],
        "annotations_dir": str(Path(annotations_dir).resolve()),
        "videos": {
            stem: {
                "path": raw["videos"][stem]["path"],
                "width": raw["videos"][stem].get("width"),
                "height": raw["videos"][stem].get("height"),
                "fps": raw["videos"][stem].get("fps"),
                "num_frames": raw["videos"][stem].get("num_frames"),
                **geometry[stem],
            }
            for stem in sorted(raw["videos"])
        },
        "items": items,
        "labels": {},
        "history": [],
    }


def create_app(reviews_path: Path, annotations_dir: Path, output: Path) -> Flask:
    output = Path(output).resolve()
    if output == Path(reviews_path).resolve():
        raise ValueError("Output must not overwrite the contact review")
    fresh = build_project(Path(reviews_path), Path(annotations_dir))
    if output.exists():
        project = json.loads(output.read_text(encoding="utf-8"))
        if project.get("review_sha256") != fresh["review_sha256"]:
            raise ValueError(
                "The contact review changed since these labels were started; "
                "use a new output path"
            )
        # Geometry is recomputed every start, so a changed analysis is visible
        # rather than frozen into a stale file. Human labels are never touched.
        project["items"] = fresh["items"]
        project["annotations_dir"] = fresh["annotations_dir"]
        project["videos"] = fresh["videos"]
    else:
        project = fresh

    app = Flask(__name__)
    lock = threading.Lock()
    decode_lock = threading.Lock()
    captures: dict[str, Any] = {}
    by_id = {item["id"]: item for item in project["items"]}

    def persist(value: dict[str, Any]) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(".tmp")
        temp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        temp.replace(output)

    @app.after_request
    def cache_policy(response):
        # State must never be served stale. A rendered contact frame is
        # immutable for its id, and caching it is what makes prefetching the
        # next few worth anything: a 4K seek and decode costs seconds.
        if request.path.startswith("/api/frame/"):
            response.headers["Cache-Control"] = "private, max-age=3600"
        else:
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def index():
        return Path(__file__).with_name("team_attribution.html").read_text(
            encoding="utf-8",
        )

    @app.get("/api/project")
    def get_project():
        with lock:
            # The geometric prediction never reaches the page before an answer
            # exists for that contact. Anything sent here can be read from the
            # browser, so it is withheld server side rather than merely hidden.
            items = []
            for item in project["items"]:
                labelled = item["id"] in project["labels"]
                public = {
                    key: item[key] for key in
                    ("id", "video_id", "frame", "contact_type", "ball",
                     "ball_status")
                }
                public["attribution"] = (
                    item["attribution"] if labelled else None
                )
                items.append(public)
            return jsonify({
                "schema": project["schema"],
                "videos": project["videos"],
                "items": items,
                "labels": project["labels"],
                "counts": {
                    "total": len(items),
                    "labelled": len(project["labels"]),
                },
            })

    @app.get("/api/frame/<path:identity>")
    def get_frame(identity: str):
        item = by_id.get(identity)
        if item is None:
            return jsonify(error="Unknown contact"), 404
        video = project["videos"][item["video_id"]]
        frame_number = item["frame"]
        with decode_lock:
            capture = captures.get(item["video_id"])
            if capture is None:
                capture = captures[item["video_id"]] = cv2.VideoCapture(
                    video["path"],
                )
            if int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) != frame_number:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, image = capture.read()
            actual = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            if not ok or actual != frame_number:
                return jsonify(error="Native frame decode failed"), 500
        canvas = _render(image, item, video)
        ok, encoded = cv2.imencode(
            ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88],
        )
        if not ok:
            return jsonify(error="Image encoding failed"), 500
        return send_file(io.BytesIO(encoded.tobytes()), mimetype="image/jpeg")

    @app.post("/api/label")
    def save_label():
        if request.headers.get("Sec-Fetch-Site", "same-origin") != "same-origin":
            return jsonify(error="Cross-site write refused"), 403
        payload = request.get_json(silent=True) or {}
        identity = payload.get("id")
        value = payload.get("team")
        if identity not in by_id:
            return jsonify(error="Unknown contact"), 400
        if value not in ("team_a", "team_b", "unsure", None):
            return jsonify(error="Invalid team"), 400
        with lock:
            if value is None:
                project["labels"].pop(identity, None)
            else:
                project["labels"][identity] = {
                    "team": value,
                    "labelled_at": _utc_now(),
                }
            project["history"].append({
                "id": identity, "team": value, "at": _utc_now(),
            })
            persist(project)
            return jsonify({
                "ok": True,
                "labelled": len(project["labels"]),
                # Revealed only now, with the answer already recorded.
                "attribution": by_id[identity]["attribution"],
            })

    @app.get("/api/agreement")
    def agreement():
        with lock:
            records = [
                {
                    "human": project["labels"][item["id"]]["team"],
                    "attribution": item["attribution"],
                }
                for item in project["items"]
                if item["id"] in project["labels"]
                and project["labels"][item["id"]]["team"] in ta.TEAMS
            ]
            return jsonify({
                "labelled": len(project["labels"]),
                "comparable": len(records),
                "methods": [
                    ta.score_against_human(records, method).to_dict()
                    for method in ("nearest_player", "ball_side", "prediction")
                ],
            })

    return app


def _render(image: np.ndarray, item: dict[str, Any], video: dict[str, Any]):
    """Frame with the divider, the two halves named, and the ball marked.

    Poses are deliberately not drawn: the label should come from the action in
    the frame, not from the same evidence the geometric prediction uses.
    """
    height, width = image.shape[:2]
    scale = RENDER_WIDTH / float(width)
    canvas = cv2.resize(image, (RENDER_WIDTH, int(round(height * scale))))
    left = video.get("net_floor_left")
    right = video.get("net_floor_right")
    if left and right:
        start = (int(round(left[0] * scale)), int(round(left[1] * scale)))
        end = (int(round(right[0] * scale)), int(round(right[1] * scale)))
        # Extend the divider across the frame; the stored points only span the
        # posts, and the side of a contact outside them still has to be read.
        if end[0] != start[0]:
            slope = (end[1] - start[1]) / (end[0] - start[0])
            start = (0, int(round(start[1] - slope * start[0])))
            end = (
                canvas.shape[1],
                int(round(end[1] + slope * (canvas.shape[1] - end[0]))),
            )
        cv2.line(canvas, start, end, (255, 0, 255), 3)
        mid_y = (start[1] + end[1]) // 2
        cv2.putText(
            canvas, "A  far side", (24, max(36, mid_y - 24)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 255), 3,
        )
        cv2.putText(
            canvas, "B  near side", (24, min(canvas.shape[0] - 16, mid_y + 56)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 0, 255), 3,
        )
    ball = item.get("ball")
    if ball:
        centre = (int(round(ball[0] * scale)), int(round(ball[1] * scale)))
        cv2.circle(canvas, centre, 26, (0, 255, 255), 3)
        cv2.line(
            canvas, (centre[0] - 44, centre[1]), (centre[0] - 30, centre[1]),
            (0, 255, 255), 3,
        )
        cv2.line(
            canvas, (centre[0] + 30, centre[1]), (centre[0] + 44, centre[1]),
            (0, 255, 255), 3,
        )
        cv2.line(
            canvas, (centre[0], centre[1] - 44), (centre[0], centre[1] - 30),
            (0, 255, 255), 3,
        )
        cv2.line(
            canvas, (centre[0], centre[1] + 30), (centre[0], centre[1] + 44),
            (0, 255, 255), 3,
        )
    else:
        cv2.putText(
            canvas, f"ball {item.get('ball_status')} - no marker",
            (24, canvas.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 200, 255), 3,
        )
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviews", type=Path,
        default=ROOT / "data/contact_spatial_gold_v1/reviews.json",
    )
    parser.add_argument(
        "--annotations-dir", type=Path,
        default=ROOT / "results/gold_pose_fsm_20261028_v1",
        help="pose-enabled analyses of the reviewed videos",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/contact_team_gold_v1/labels.json",
    )
    parser.add_argument("--port", type=int, default=5057)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)

    app = create_app(args.reviews, args.annotations_dir, args.output)
    print(f"[team-annotator] labels -> {args.output}")
    print(f"[team-annotator] http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
