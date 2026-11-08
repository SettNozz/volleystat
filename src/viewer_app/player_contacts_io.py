"""Loading paired player-contact inputs and roster portrait references.

Moved verbatim from viewer.py, which re-exports these names.
"""

import json
import math
from functools import lru_cache
from pathlib import Path


from src.tracking.player_contacts import build_player_contacts, source_fingerprint
from src.tracking.contact_author_experiments import experiment_path, load_experiment


def _roster_portrait_references(payload):
    seed = (payload.get("player_identity") or {}).get("seed") or {}
    keys = dict(seed.get("assignments") or {})
    keys.update((item["key"], item["player_id"]) for item in seed.get("introductions") or [])
    by_frame = {row["f"]: row for row in payload.get("frames") or []}
    references = {}
    for key, player in keys.items():
        try:
            frame, index = map(int, key.split(":"))
            if frame < 0 or index < 0:
                continue
            box = tuple(by_frame[frame]["poses"][index]["bb"])
            if len(box) == 4 and all(math.isfinite(value) for value in box):
                references.setdefault(player, [(frame, box)])
        except (ValueError, KeyError, IndexError, TypeError):
            continue
    # Sample the tracked roster across the match so one bad initial frame
    # does not decide every portrait. Only poses on the roster's court side qualify.
    identity = payload.get("player_identity") or {}
    target = {"a": "a", "b": "b", "team_a": "a", "team_b": "b"}.get(identity.get("side"))
    interval = max(1, (max(by_frame, default=0) + 8) // 8)
    samples = {}
    for frame, row in by_frame.items():
        for pose in row.get("poses") or []:
            player = pose.get("player_id")
            if not target or pose.get("team") not in (target, f"team_{target}") or not player:
                continue
            box = tuple(pose.get("bb") or ())
            if len(box) != 4 or not all(math.isfinite(value) for value in box):
                continue
            width, height = box[2] - box[0], box[3] - box[1]
            if width <= 0 or height <= 0:
                continue
            key = (player, frame // interval)
            if key not in samples or width * height > samples[key][0]:
                samples[key] = (width * height, (frame, box))
    for (player, _), (_, reference) in samples.items():
        choices = references.setdefault(player, [])
        if reference not in choices:
            choices.append(reference)
    return references


@lru_cache(maxsize=2)
def _player_contact_inputs(stats_path, ann_path, stats_stamp, ann_stamp, experiment_stamp=None):
    """Cache derived proposals, not the large frame/pose payload."""
    stats_bytes, ann_bytes = Path(stats_path).read_bytes(), Path(ann_path).read_bytes()
    stats = json.loads(stats_bytes)
    source = source_fingerprint(stats_bytes, ann_bytes)
    payload = json.loads(ann_bytes)
    references = _roster_portrait_references(payload)
    # Older contact reports omit explicit serve faults. Attribute these
    # separately for statistics without changing saved reports or reviews.
    faults = [(index, dict(event, type="serve")) for index, event in enumerate(stats.get("all_events") or [])
              if event.get("type") == "serve_fault"]
    fault_contacts = []
    if faults:
        fault_contacts = build_player_contacts(
            [event for _, event in faults], payload["frames"], payload["fps"], payload.get("player_identity"),
        )["contacts"]
        for contact in fault_contacts:
            contact["event_index"] = faults[contact["event_index"]][0]
    if experiment_stamp is not None:
        bundle = load_experiment(experiment_path(stats_path), source, stats)
        return bundle["reports"][bundle["selected_method"]], source, bundle, references, fault_contacts
    if stats.get("video") != payload.get("video") or float(stats["fps"]) != float(payload["fps"]):
        raise ValueError("Stats and annotations must belong to the same analysis")
    report = payload.get("player_contacts")
    if report:
        from src.tracking.player_contacts import validate_saved_report
        validate_saved_report(report, stats)
    else:
        report = build_player_contacts(stats.get("all_events") or [], payload["frames"],
                                       payload["fps"], payload.get("player_identity"))
    return report, source, None, references, fault_contacts


def _load_player_contact_inputs(job):
    stats_path, ann_path = Path(job["stats_path"]), Path(job["annotations_path"])

    def stamp(path):
        info = path.stat()
        return info.st_mtime_ns, info.st_size

    experiment = experiment_path(stats_path)
    return _player_contact_inputs(
        str(stats_path.resolve()), str(ann_path.resolve()), stamp(stats_path), stamp(ann_path),
        stamp(experiment) if experiment.exists() else None,
    )
