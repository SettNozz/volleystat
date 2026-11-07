"""Read-only author-error review over frozen human labels and held-out predictions."""
import argparse
import hashlib
import io
import json
from pathlib import Path

from flask import Flask, jsonify, send_file

from .contact_author_annotator import portrait_bytes
from .contact_author_dataset import sha


def create_app(evaluation_dir):
    directory = Path(evaluation_dir).resolve()
    report = json.loads((directory/"comparison.json").read_bytes())
    dataset = Path(report["dataset"])
    if sha(dataset/"manifest.json")!=report["dataset_manifest_sha256"]:
        raise ValueError("Benchmark dataset changed")
    manifest = json.loads((dataset/"manifest.json").read_bytes())
    if sha(dataset/"human_labels.json")!=manifest["snapshot_sha256"] or sha(directory/"predictions.json")!=report["predictions_sha256"]:
        raise ValueError("Human snapshot or predictions changed")
    snapshot = json.loads((dataset/"human_labels.json").read_bytes())
    records = json.loads((directory/"predictions.json").read_bytes())
    truth = {c["id"]:c for c in snapshot["contacts"]}
    if set(truth)!={r["id"] for r in records}: raise ValueError("Prediction queue differs from human snapshot")
    needed = {v:set() for v in snapshot["videos"]}
    for row in records:
        c = truth[row["id"]]
        if (row["frame"],row["video_id"],row["truth"])!=(c["frame"],c["video_id"],c["author"]["player_id"] or "other_team"):
            raise ValueError("Prediction row changed human truth")
        needed[row["video_id"]].add(row["frame"])
        for p in row["predictions"].values():
            if p["label_source"]!="derived" or p["verified"] is not False: raise ValueError("Predictions must remain derived")
            if p.get("actor"): needed[row["video_id"]].add(p["actor"]["frame"])
    evidence = {}
    for video,metadata in snapshot["videos"].items():
        path = Path(metadata["annotations_source"]["path"]); raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=metadata["annotations_source"]["sha256"]:
            raise ValueError("Cached source annotations changed")
        data = json.loads(raw)
        evidence[video] = {r["f"]:{k:r.get(k) for k in ("f","poses","ball","ball_source","ball_interp","ball_out_of_frame")}
                           for r in data["frames"] if r["f"] in needed[video]}
        del data,raw
    for row in records:
        for p in row["predictions"].values():
            actor = p.get("actor")
            if actor is None: continue
            poses = evidence[row["video_id"]][actor["frame"]]["poses"] or []
            i = actor["pose_index"]
            if not 0<=i<len(poses) or poses[i].get("id")!=actor["raw_track_id"] or poses[i]["bb"]!=actor.get("bbox",poses[i]["bb"]):
                raise ValueError("Selected actor differs from cached source pose")
    app = Flask(__name__)

    def metadata_for(video):
        if video not in snapshot["videos"]: raise ValueError("Unknown video")
        metadata = snapshot["videos"][video]; source = metadata["video_source"]
        stat = Path(source["path"]).stat()
        if stat.st_size!=source["size"] or stat.st_mtime_ns!=source["mtime_ns"]: raise ValueError("Original video changed")
        return metadata

    @app.errorhandler(ValueError)
    def invalid(error): return jsonify(error=str(error)),400

    @app.get("/")
    def index(): return send_file(Path(__file__).with_name("contact_author_benchmark.html"))

    @app.get("/api/report")
    def data():
        return jsonify(report=report,records=records,rosters=snapshot["rosters"],
                       videos={v:{k:m[k] for k in ("fps","width","height","num_frames")} for v,m in snapshot["videos"].items()})

    @app.get("/video/<video>")
    def video(video): return send_file(metadata_for(video)["video_source"]["path"],conditional=True)

    @app.get("/api/frame/<video>/<int:frame>")
    def frame(video,frame):
        metadata_for(video)
        return jsonify(evidence[video].get(frame,{"f":frame,"poses":[],"ball":None}))

    @app.get("/photo/<video>/<player_id>")
    def photo(video,player_id):
        metadata = metadata_for(video)
        player = next((p for p in snapshot["rosters"][video]["players"] if p["player_id"]==player_id),None)
        if player is None: raise ValueError("Unknown roster reference")
        ref = player["references"][0]
        key = hashlib.sha256(json.dumps([metadata["video_source"]["sha256"],ref],sort_keys=True).encode()).hexdigest()
        cache = directory/"review_photos"/f"{key}.jpg"
        if not cache.exists():
            raw = portrait_bytes(metadata,ref); cache.parent.mkdir(exist_ok=True)
            try:
                with cache.open("xb") as stream: stream.write(raw)
            except FileExistsError: pass
        return send_file(io.BytesIO(cache.read_bytes()),mimetype="image/jpeg")

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation",type=Path,required=True)
    parser.add_argument("--port",type=int,default=5067)
    args = parser.parse_args()
    create_app(args.evaluation).run(host="127.0.0.1",port=args.port,debug=False,threaded=True)


if __name__=="__main__": main()
