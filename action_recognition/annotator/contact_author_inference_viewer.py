"""Read-only predictions on an unlabelled source, with original cached poses."""
import argparse
import io
import json
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from .contact_author_annotator import portrait_bytes
from .contact_author_dataset import sha


def create_app(directory):
    directory=Path(directory).resolve();report=json.loads((directory/"report.json").read_bytes())
    if report["schema"]!="contact_author_unlabelled_inference_v1":raise ValueError("Unknown inference schema")
    dataset=Path(report["dataset"]);manifest=json.loads((dataset/"manifest.json").read_bytes())
    if sha(dataset/"manifest.json")!=report["dataset_manifest_sha256"] or sha(directory/"predictions.json")!=report["predictions_sha256"]:
        raise ValueError("Inference artifacts changed")
    snapshot_file=manifest["snapshot_file"]
    if sha(dataset/snapshot_file)!=manifest["snapshot_sha256"]:raise ValueError("Inference inputs changed")
    snapshot=json.loads((dataset/snapshot_file).read_bytes());records=json.loads((directory/"predictions.json").read_bytes())
    contacts={c["id"]:c for c in snapshot["contacts"]}
    if set(contacts)!={r["id"] for r in records}:raise ValueError("Prediction queue differs from input events")
    evidence={}
    for video,metadata in snapshot["videos"].items():
        if sha(metadata["annotations_source"]["path"])!=metadata["annotations_source"]["sha256"]:raise ValueError("Source observations changed")
        data=json.loads(Path(metadata["annotations_source"]["path"]).read_bytes())
        evidence[video]={r["f"]:{k:r.get(k) for k in ("f","poses","ball","ball_out_of_frame","ball_source")} for r in data["frames"]}
        del data
    for row in records:
        contact=contacts[row["id"]]
        if row["truth"] is not None or (row["video_id"],row["frame"],row["type"])!=(contact["video_id"],contact["frame"],contact["type"]):
            raise ValueError("Predictions must keep original model events and no human truth")
        for p in row["predictions"].values():
            if p["label_source"]!="derived" or p["verified"] is not False:raise ValueError("Predictions cannot become human labels")
            a=p.get("actor")
            if a:
                pose=evidence[row["video_id"]][a["frame"]]["poses"][a["pose_index"]]
                if pose["id"]!=a["raw_track_id"] or pose["bb"]!=a["bbox"]:raise ValueError("Actor differs from saved pose")
    app=Flask(__name__);portraits={}

    def metadata(video):
        if video not in snapshot["videos"]:raise ValueError("Unknown video")
        m=snapshot["videos"][video];stat=Path(m["video_source"]["path"]).stat()
        if stat.st_size!=m["video_source"]["size"] or stat.st_mtime_ns!=m["video_source"]["mtime_ns"]:raise ValueError("Original video changed")
        return m

    @app.errorhandler(ValueError)
    def invalid(error):return jsonify(error=str(error)),400

    @app.get("/")
    def index():return send_file(Path(__file__).with_name("contact_author_inference.html"))

    @app.get("/api/report")
    def report_data():
        return jsonify(report=report,records=records,rosters=snapshot["rosters"],
                       videos={v:{k:m[k] for k in ("fps","width","height","num_frames")} for v,m in snapshot["videos"].items()})

    @app.get("/video/<video>")
    def original(video):return send_file(metadata(video)["video_source"]["path"],conditional=True)

    @app.get("/api/frames/<video>")
    def frames(video):
        m=metadata(video);start=max(0,int(request.args.get("start",0)));end=min(m["num_frames"]-1,int(request.args.get("end",start)))
        if end<start or end-start>180:raise ValueError("Request at most 181 saved frames")
        return jsonify(frames=[evidence[video][f] for f in range(start,end+1) if f in evidence[video]])

    @app.get("/photo/<video>/<player_id>")
    def photo(video,player_id):
        m=metadata(video);player=next((p for p in snapshot["rosters"][video]["players"] if p["player_id"]==player_id),None)
        if player is None:raise ValueError("Unknown roster ID")
        key=(video,player_id)
        if key not in portraits:portraits[key]=portrait_bytes(m,player["references"][0])
        return send_file(io.BytesIO(portraits[key]),mimetype="image/jpeg")

    return app


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--predictions",type=Path,required=True);parser.add_argument("--port",type=int,default=5068)
    args=parser.parse_args();create_app(args.predictions).run(host="127.0.0.1",port=args.port,debug=False,threaded=True)


if __name__=="__main__":main()
