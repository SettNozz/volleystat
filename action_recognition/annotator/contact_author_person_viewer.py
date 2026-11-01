"""Read-only review of person-ranker contact authors on an unlabelled source video."""
import argparse
import io
import json
from pathlib import Path

from flask import Flask, jsonify, request, send_file

from .contact_author_annotator import portrait_bytes
from .contact_author_dataset import sha


def create_app(directory):
    directory = Path(directory).resolve(); report = json.loads((directory/"report.json").read_bytes())
    if report["schema"] != "contact_author_person_inference_v1":
        raise ValueError("Unknown person-ranker inference schema")
    dataset = Path(report["dataset"]); manifest = json.loads((dataset/"manifest.json").read_bytes())
    if sha(dataset/"manifest.json") != report["dataset_manifest_sha256"] or sha(directory/"predictions.json") != report["predictions_sha256"]:
        raise ValueError("Inference artifacts changed")
    if sha(dataset/manifest["snapshot_file"]) != manifest["snapshot_sha256"]:
        raise ValueError("Inference inputs changed")
    snapshot = json.loads((dataset/manifest["snapshot_file"]).read_bytes()); records = json.loads((directory/"predictions.json").read_bytes())
    contacts = {c["id"]: c for c in snapshot["contacts"]}
    if set(contacts) != {r["id"] for r in records}:
        raise ValueError("Prediction queue differs from input events")
    evidence = {}
    for video, meta in snapshot["videos"].items():
        if sha(meta["annotations_source"]["path"]) != meta["annotations_source"]["sha256"]:
            raise ValueError("Source observations changed")
        data = json.loads(Path(meta["annotations_source"]["path"]).read_bytes())
        evidence[video] = {r["f"]: {"f": r["f"], "ball": r.get("ball"), "ball_out_of_frame": r.get("ball_out_of_frame"),
                                    "poses": [{k: p.get(k) for k in ("bb", "kp", "id", "team", "player_id")} for p in r.get("poses") or []]}
                           for r in data["frames"]}
        del data
    for row in records:
        contact = contacts[row["id"]]
        if row["truth"] is not None or (row["video_id"], row["frame"], row["type"]) != (contact["video_id"], contact["frame"], contact["type"]):
            raise ValueError("Predictions must keep original model events and no human truth")
        for p in row["predictions"].values():
            if p.get("label_source") != "derived" or p.get("verified") is not False:
                raise ValueError("Predictions cannot become human labels")
            actor = p.get("actor")
            if actor:
                pose = evidence[row["video_id"]][actor["frame"]]["poses"][actor["pose_index"]]
                if pose["id"] != actor["raw_track_id"] or pose["bb"] != actor["bbox"]:
                    raise ValueError("Evidence pose differs from saved annotations")
    app = Flask(__name__); portraits = {}

    def metadata(video):
        if video not in snapshot["videos"]:
            raise ValueError("Unknown video")
        meta = snapshot["videos"][video]; stat = Path(meta["video_source"]["path"]).stat()
        if stat.st_size != meta["video_source"]["size"] or stat.st_mtime_ns != meta["video_source"]["mtime_ns"]:
            raise ValueError("Original video changed")
        return meta

    @app.errorhandler(ValueError)
    def invalid(error):
        return jsonify(error=str(error)), 400

    @app.get("/")
    def index():
        return send_file(Path(__file__).with_name("contact_author_person_viewer.html"))

    @app.get("/api/report")
    def report_data():
        return jsonify(report=report, records=records, rosters=snapshot["rosters"],
                       videos={v: {k: m[k] for k in ("fps", "width", "height", "num_frames")} for v, m in snapshot["videos"].items()})

    @app.get("/video/<video>")
    def original(video):
        return send_file(metadata(video)["video_source"]["path"], conditional=True)

    @app.get("/api/frames/<video>")
    def frames(video):
        meta = metadata(video); start = max(0, int(request.args.get("start", 0)))
        end = min(meta["num_frames"]-1, int(request.args.get("end", start)))
        if end < start or end-start > 180:
            raise ValueError("Request at most 181 saved frames")
        return jsonify(frames=[evidence[video][f] for f in range(start, end+1) if f in evidence[video]])

    @app.get("/photo/<video>/<player_id>")
    def photo(video, player_id):
        meta = metadata(video)
        player = next((p for p in snapshot["rosters"][video]["players"] if p["player_id"] == player_id), None)
        if player is None:
            raise ValueError("Unknown roster ID")
        if (video, player_id) not in portraits:
            portraits[video, player_id] = portrait_bytes(meta, player["references"][0])
        return send_file(io.BytesIO(portraits[video, player_id]), mimetype="image/jpeg")

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--port", type=int, default=5069)
    args = parser.parse_args()
    create_app(args.predictions).run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
