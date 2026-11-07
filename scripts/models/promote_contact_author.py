"""Install the pinned winning author model, or export identity-only photo setup."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.tracking.contact_author_runtime import ContactAuthorModel, MODEL_SHA256, METHOD, sha


def install(source, output):
    model = ContactAuthorModel(source)
    target = output/"model.joblib"
    if target.exists() and sha(target) != MODEL_SHA256:
        raise FileExistsError("Installed author version differs; choose a fresh version directory")
    output.mkdir(parents=True, exist_ok=True)
    if not target.exists(): shutil.copy2(source, target)
    deployment = {"schema": "contact_author_deployment_v1", "version": "expanded_20261106_v1",
        "promoted_date": "2026-11-07", "method": METHOD, "model_sha256": model.sha256,
        "training_artifact": str(source.resolve()), "evaluation": "docs/experiments/contact_expanded_training_20261106_v1.md",
        "evaluation_scope": "development whole-source held-out folds; human photo roster; known contacts",
        "near_correct": 470, "near_count": 565, "new7_near_correct": 307, "new7_near_count": 381,
        "final_fit_heldout_score": None, "event_models_changed": False}
    path = output/"deployment.json"
    if not path.exists(): path.write_text(json.dumps(deployment, indent=2), encoding="utf-8")
    return {"model": str(target.resolve()), "sha256": sha(target)}


def export_roster(project, video_id, output):
    from action_recognition.annotator.contact_author_dataset import AnnotationStore, write_new
    if output.exists(): raise FileExistsError("Choose a fresh photo-roster filename")
    store = AnnotationStore(project); store.check_sources(); state = store.load()
    metadata = store.project["videos"][video_id]; roster = state["rosters"][video_id]
    if not roster["confirmed"]: raise ValueError("Confirm photo identities first")
    result = {"schema": "contact_author_photo_roster_v1", "side": metadata["target_side"],
        "video_sha256": metadata["video_source"]["sha256"], "players": [
            {"player_id": p["player_id"], "references": [{"frame": r["frame"], "bbox": deepcopy(r["bbox"])} for r in p["references"]]}
            for p in roster["players"]], "label_source": "human", "verified": True, "scope": "identity_references_only",
        "max_on_court": min(6, len(roster["players"])), "project_sha256": store.fingerprint,
        "identity_revision": state["revision"]}
    write_new(output, result)
    return {"output": str(output.resolve()), "people": len(result["players"]), "contact_author_targets_exported": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    added = sub.add_parser("install"); added.add_argument("--source", type=Path, default=ROOT/"results/contact_expanded_training_20261106_v1/final_author/model.joblib")
    added.add_argument("--output", type=Path, default=ROOT/"models/contact_author/expanded_20261106_v1")
    roster = sub.add_parser("export-roster"); roster.add_argument("--project", type=Path, required=True)
    roster.add_argument("--video-id", required=True); roster.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(install(args.source, args.output) if args.command == "install" else
        export_roster(args.project, args.video_id, args.output), ensure_ascii=False, indent=2))
