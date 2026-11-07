"""Human spatial-contact review, built on the full-video annotator's sources.
Run: python -m action_recognition.annotator.contact_spatial_annotator
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import threading
from pathlib import Path
import cv2
from flask import Flask, jsonify, request, send_file
from .full_video_annotator import _source_videos, _utc_now
from action_recognition.event_clips.config import collapse_labels

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ("serve", "reception", "pass", "attack")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def build_project(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    videos = {v["video_id"]: v for v in _source_videos(manifest)}
    sources, humans = {}, {}
    for stem in videos:
        path = ROOT / "data/full_video_gold" / stem / "annotations_v1.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        humans[stem] = {a["annotation_id"]: a for a in data["annotations"] if a["video_id"] == stem}
        sources[stem] = {"path": str(path), "sha256": sha(path)}
        cap = cv2.VideoCapture(videos[stem]["path"])
        if not cap.isOpened():
            raise ValueError(f"Cannot open original video: {videos[stem]['path']}")
        videos[stem].update(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
    events = {}
    for clip in manifest["clips"]:
        label = collapse_labels(clip.get("event_types", [clip.get("primary_event_type")]), clip.get("fsm_event_type"))
        if label not in CLASSES:
            continue
        stem, frame = clip["source_stem"], int(clip["contact_frame"])
        key = f"{stem}:{frame}:{label}"
        event = events.setdefault(key, dict(id=key, video_id=stem, original_frame=frame, label=label, annotation_ids=[], raw_labels=[]))
        for aid in clip.get("annotation_ids", []):
            if aid not in humans[stem] or humans[stem][aid].get("label_source") != "human" or humans[stem][aid]["frame"] != frame:
                raise ValueError(f"Missing human annotation {aid}")
            if aid not in event["annotation_ids"]:
                event["annotation_ids"].append(aid)
                event["raw_labels"].append(humans[stem][aid]["event_type"])
        if not event["annotation_ids"]:
            raise ValueError(f"No human provenance for {key}")
    # Preserve extra authoritative IDs for exact same-frame/class duplicates.
    for stem, annotations in humans.items():
        for aid, annotation in annotations.items():
            label = collapse_labels([annotation["event_type"]])
            if label is None:
                continue
            key = f"{stem}:{annotation['frame']}:{label}"
            if key not in events:
                raise ValueError(f"Human contact missing from active manifest: {key}")
            if annotation.get("label_source") != "human":
                raise ValueError(f"Non-human provenance: {aid}")
            if aid not in events[key]["annotation_ids"]:
                events[key]["annotation_ids"].append(aid)
                events[key]["raw_labels"].append(annotation["event_type"])
    return dict(schema="contact_spatial_review_v1", created_at=_utc_now(), manifest_path=str(manifest_path), manifest_sha256=sha(manifest_path), sources=sources, videos=videos, events=sorted(events.values(), key=lambda e:(e["video_id"], e["original_frame"], e["label"])), reviews={}, history=[])

def validate_review(payload, event, video):
    frame = payload.get("frame")
    if isinstance(frame, bool) or not isinstance(frame, int) or not 0 <= frame < video["num_frames"]:
        raise ValueError("Invalid native frame")
    status = payload.get("status")
    if status not in ("visible", "occluded", "out_of_frame", "uncertain", "wrong_event"):
        raise ValueError("Choose contact visibility")
    timing = payload.get("timing")
    if timing not in ("clear", "uncertain"):
        raise ValueError("Confirm temporal precision")
    label = payload.get("label")
    if label not in CLASSES:
        raise ValueError("Invalid event class")
    reviewer = str(payload.get("reviewer", "")).strip()[:80]
    if not reviewer:
        raise ValueError("Enter reviewer name")
    xy = payload.get("xy")
    if status == "visible":
        if not isinstance(xy, list) or len(xy) != 2 or any(isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) for v in xy):
            raise ValueError("Click the ball centre")
        if not (0 <= xy[0] < video["width"] and 0 <= xy[1] < video["height"]):
            raise ValueError("Coordinates outside original image")
        xy = [round(float(v), 3) for v in xy]
    else:
        xy = None
    return dict(frame=frame, label=label, status=status, timing=timing, xy=xy, xy_normalized=[xy[0]/video["width"],xy[1]/video["height"]] if xy else None, width=video["width"], height=video["height"], reviewer=reviewer, note=str(payload.get("note", ""))[:2000], verified=True, ball_out_of_frame=status=="out_of_frame", spatial_valid=status=="visible" and timing=="clear", temporal_valid=status!="wrong_event" and timing=="clear", updated_at=_utc_now())

def create_app(manifest_path, output):
    output = Path(output).resolve()
    fresh = build_project(manifest_path)
    source_paths = {Path(s["path"]).resolve() for s in fresh["sources"].values()}
    if output in source_paths or output == Path(manifest_path).resolve():
        raise ValueError("Output must not overwrite source labels")
    if output.exists():
        project = json.loads(output.read_text(encoding="utf-8"))
        if project["manifest_sha256"] != fresh["manifest_sha256"] or project["sources"] != fresh["sources"]:
            raise ValueError("Source labels changed; use a new review output")
    else:
        project = fresh
    app = Flask(__name__)
    lock, decode_lock = threading.Lock(), threading.Lock()
    events = {e["id"]:e for e in project["events"]}
    captures = {}
    def persist(value):
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(".tmp")
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(output)
    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        return response
    @app.get("/")
    def index():
        return (Path(__file__).with_name("contact_spatial.html")).read_text(encoding="utf-8")
    @app.get("/api/project")
    def get_project():
        with lock:
            return jsonify({k:v for k,v in project.items() if k != "history"})
    @app.get("/api/frame/<stem>/<int:frame>")
    def get_frame(stem, frame):
        video = project["videos"].get(stem)
        if not video or not 0 <= frame < video["num_frames"]:
            return jsonify(error="Frame outside video"), 400
        with decode_lock:
            cap = captures.setdefault(stem, None)
            if cap is None:
                cap = captures[stem] = cv2.VideoCapture(video["path"])
            if int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            actual = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            if not ok or actual != frame:
                return jsonify(error="Native frame decode failed; no image saved"), 500
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 96])
        if not ok:
            return jsonify(error="Image encoding failed"), 500
        return send_file(io.BytesIO(encoded.tobytes()), mimetype="image/jpeg")
    @app.post("/api/review")
    def save_review():
        # Same-origin local tool: reject cross-site writes.
        if request.headers.get("Sec-Fetch-Site") == "cross-site":
            return jsonify(error="Cross-site write rejected"), 403
        data = request.get_json(silent=True) or {}
        event = events.get(data.get("id"))
        if event is None:
            return jsonify(error="Unknown event"), 400
        try:
            review = validate_review(data, event, project["videos"][event["video_id"]])
            with lock:
                old = project["reviews"].get(event["id"])
                if data.get("revision", 0) != (old or {}).get("revision", 0):
                    return jsonify(error="This contact changed in another tab. Reload before saving."), 409
                review["revision"] = (old or {}).get("revision", 0) + 1
                updated = dict(project, reviews=dict(project["reviews"]), history=list(project["history"]))
                updated["reviews"][event["id"]] = review
                updated["history"].append(dict(id=event["id"], before=old, after=review))
                persist(updated)
                project.update(updated)
            return jsonify(review)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
    @app.get("/api/export")
    def export():
        with lock:
            rows = []
            for event in project["events"]:
                review = project["reviews"].get(event["id"])
                if review:
                    rows.append(dict(event, **review))
            data = dict(schema=project["schema"], manifest_sha256=project["manifest_sha256"], sources=project["sources"], coordinate_system="original_image_pixels; origin top-left; zero-based native frames", events=rows)
        return send_file(io.BytesIO(json.dumps(data, ensure_ascii=False, indent=2).encode()), mimetype="application/json", as_attachment=True, download_name="contact_spatial_export.json")
    app.config["REVIEW_PROJECT"] = project
    return app

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT/"data/event_clips_three_full_gold_causal_120f_v1/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT/"data/contact_spatial_gold_v1/reviews.json")
    parser.add_argument("--port", type=int, default=5062)
    args = parser.parse_args()
    app = create_app(args.manifest, args.output)
    print(f"Contact review: http://127.0.0.1:{args.port} | {len(app.config['REVIEW_PROJECT']['events'])} contacts", flush=True)
    app.run(host="127.0.0.1", port=args.port, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()
