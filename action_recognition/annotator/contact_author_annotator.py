"""Local human contact-author annotation UI; no model inference or label promotion."""
import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, send_file

from .contact_author_dataset import AnnotationStore, Conflict


def progress(project,state):
    result = {}
    for video in project["videos"]:
        rows = [c for c in project["contacts"] if c["video_id"]==video]
        counts = Counter(state["labels"][c["id"]]["decision"] for c in rows if c["id"] in state["labels"])
        result[video] = {"total":len(rows),"reviewed":sum(counts.values()),"unlabelled":len(rows)-sum(counts.values()),
                         "decisions":dict(counts)}
    return result


def portrait_bytes(metadata,reference):
    cap = cv2.VideoCapture(metadata["video_source"]["path"])
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES,reference["frame"])
        ok,frame = cap.read()
    finally: cap.release()
    if not ok: raise ValueError("Cannot decode reference frame")
    x1,y1,x2,y2 = reference["bbox"]
    h,w = frame.shape[:2]
    x1,y1,x2,y2 = max(0,math.floor(x1)),max(0,math.floor(y1)),min(w,math.ceil(x2)),min(h,math.ceil(y2))
    crop = frame[y1:y2,x1:x2]
    if crop.size==0: raise ValueError("Empty player reference")
    scale = min(1.,320/crop.shape[0],240/crop.shape[1])
    crop = cv2.resize(crop,(max(1,round(crop.shape[1]*scale)),max(1,round(crop.shape[0]*scale))))
    ok,encoded = cv2.imencode(".jpg",crop,[cv2.IMWRITE_JPEG_QUALITY,88])
    if not ok: raise ValueError("Cannot encode reference image")
    return encoded.tobytes()


def create_app(project_dir,verify_full=True):
    store = AnnotationStore(project_dir)
    store.initialize(); store.check_sources(full=verify_full)
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 64*1024
    app.config["AUTHOR_STORE"] = store

    def snapshot(state=None):
        state = store.load() if state is None else state
        return {"state":state,"progress":progress(store.project,state)}

    @app.errorhandler(Conflict)
    def conflict(error): return jsonify(error=str(error)),409

    @app.errorhandler(ValueError)
    def invalid(error): return jsonify(error=str(error)),400

    @app.get("/")
    def index(): return send_file(Path(__file__).with_name("contact_author.html"))

    @app.get("/contact_author.js")
    def javascript(): return send_file(Path(__file__).with_name("contact_author.js"))

    @app.get("/api/project")
    def project():
        store.check_sources()
        videos = {key:{k:v[k] for k in ("video_id","name","fps","width","height","num_frames","target_side","contact_count")}
                  for key,v in store.project["videos"].items()}
        return jsonify(videos=videos,contacts=store.project["contacts"],source_scope=store.project["source_scope"],
                       evaluation_scope=store.project["evaluation_scope"],**snapshot())

    @app.get("/video/<video_id>")
    def video(video_id):
        if video_id not in store.project["videos"]: raise ValueError("Unknown video")
        store.check_sources()
        return send_file(store.project["videos"][video_id]["video_source"]["path"],conditional=True)

    @app.get("/portrait/<video_id>/<player_id>/<int:view>")
    def portrait(video_id,player_id,view):
        store.check_sources()
        roster = store.load()["rosters"].get(video_id)
        if roster is None: raise ValueError("Unknown video")
        player = next((p for p in roster["players"] if p["player_id"]==player_id),None)
        if player is None or not 0<=view<len(player["references"]): raise ValueError("Unknown player reference")
        reference = player["references"][view]; metadata = store.project["videos"][video_id]
        key = hashlib.sha256(json.dumps([metadata["video_source"]["sha256"],reference],sort_keys=True).encode()).hexdigest()
        cache = store.directory/"portraits"/f"{key}.jpg"
        if not cache.exists():
            raw = portrait_bytes(metadata,reference)
            cache.parent.mkdir(exist_ok=True)
            try:
                with cache.open("xb") as stream: stream.write(raw)
            except FileExistsError: pass
        return send_file(io.BytesIO(cache.read_bytes()),mimetype="image/jpeg",max_age=0)

    def body():
        payload = request.get_json()
        if not isinstance(payload,dict): raise ValueError("Expected a JSON object")
        return payload

    @app.post("/api/label")
    def label(): return jsonify(**snapshot(store.label(body())))

    @app.post("/api/roster")
    def roster(): return jsonify(**snapshot(store.roster_action(body())))

    @app.get("/api/export")
    def export():
        value = store.export()
        return send_file(io.BytesIO(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode()),
                         mimetype="application/json",as_attachment=True,
                         download_name=f"contact_authors_r{value['revision']:06d}.json")

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project",type=Path,required=True)
    parser.add_argument("--port",type=int,default=5066)
    args = parser.parse_args()
    app = create_app(args.project)
    app.run(host="127.0.0.1",port=args.port,debug=False,threaded=True)


if __name__=="__main__": main()
