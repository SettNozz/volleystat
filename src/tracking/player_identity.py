"""Experimental roster association over saved detections; no pipeline mutation.

Track IDs identify local detector tracks. Player IDs identify a manually
initialised roster. Unknown observations stay unknown; IDs are never recycled.
This pilot assumes one court side and no substitutions/side changes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

SCHEMA = "player_identity_pilot_v1"
METHODS = ("raw_track", "motion", "motion_appearance")


def overlap(a: list[float], b: list[float]) -> float:
    """Intersection over smaller box: flag crops contaminated by another player."""
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(
        0., min(a[3], b[3]) - max(a[1], b[1]))
    areas = [(p[2] - p[0]) * (p[3] - p[1]) for p in (a, b)]
    return intersection / max(min(areas), 1.)


def appearance(frame: np.ndarray, bbox: list[float]) -> list[float] | None:
    """Three HSV body bands, including brightness for black/white clothing.

    This is a cheap appearance control, not a learned person-ReID embedding.
    Avoid face/feet and most background; reject off-image/very small crops.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    left, right = max(0, round(x1 + .2*w)), min(frame.shape[1], round(x2 - .2*w))
    top, bottom = max(0, round(y1 + .18*h)), min(frame.shape[0], round(y1 + .85*h))
    if right-left < 5 or bottom-top < 15:
        return None
    crop = cv2.resize(frame[top:bottom, left:right], (32, 96))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    values = []
    for band in np.array_split(hsv, 3):
        hist = cv2.calcHist([band], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
        values.extend(np.sqrt(hist.ravel() / max(float(hist.sum()), 1.)).tolist())
    return values


def appearance_distance(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    a, b = np.asarray(a), np.asarray(b)
    return float(np.clip(1. - np.dot(a, b) / max(np.linalg.norm(a)*np.linalg.norm(b), 1e-9), 0., 1.))


def centre(obs: dict) -> np.ndarray:
    x1, y1, x2, y2 = obs["bb"]
    # Bbox centre is more stable than ankles in jumps and net occlusion.
    return np.asarray([(x1+x2)/2, (y1+y2)/2], dtype=float)


def choose_seed(frames: list[dict], roster_size: int) -> dict:
    complete = [f for f in frames if len(f["players"]) == roster_size]
    if not complete:
        raise ValueError("No complete roster frame; choose another analysis/window")
    # Prefer an early frame, then minimise contamination among the first 5 s.
    early = complete[:50]
    return min(early, key=lambda f: sum(p["occlusion"] for p in f["players"]))


def default_seed(frames: list[dict], roster_size: int) -> dict:
    frame = choose_seed(frames, roster_size)
    ordered = sorted(frame["players"], key=lambda p: centre(p)[0])
    return {"frame": frame["f"], "assignments": {p["key"]: f"P{i+1}" for i, p in enumerate(ordered)},
            "label_source": "automatic_initialisation", "verified": False}


@dataclass
class State:
    obs: dict
    frame: int
    velocity: np.ndarray
    gallery: list


def run_method(frames: list[dict], seed: dict, fps: float, method: str,
               max_gap_seconds: float = 3., ambiguity_margin: float = .08) -> dict:
    """Forward/backward one-to-one assignment from one seed frame.

    Rejected assignments do not update appearance or motion history. Gallery
    updates exclude overlapped detections. Gap expiry returns unknown even
    when clothing matches: this pilot cannot establish identity after long exits.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if not np.isfinite(fps) or fps <= 0 or max_gap_seconds <= 0:
        raise ValueError("FPS and max gap must be positive")
    seed_frame = next((f for f in frames if f["f"] == seed["frame"]), None)
    if seed_frame is None:
        raise ValueError("Seed frame missing from observations")
    by_key = {p["key"]: p for p in seed_frame["players"]}
    labels = seed["assignments"]
    if set(labels) != set(by_key) or any(not isinstance(v, str) or not v for v in labels.values()):
        raise ValueError("Seed must label every visible roster detection")
    if len(set(labels.values())) != len(labels):
        raise ValueError("Seed player IDs must be unique")
    output = {}

    def record(obs: dict, pid: str | None, reason: str, cost: float | None = None):
        output[obs["key"]] = {"player_id": pid, "reason": reason, "cost": cost,
                              "label_source": "model", "verified": False}

    for key, pid in labels.items():
        record(by_key[key], pid, "seed")
    seed_raw_counts = Counter(str(p["raw_id"]) for p in by_key.values() if p["raw_id"] is not None)
    raw_map = {str(by_key[k]["raw_id"]): pid for k, pid in labels.items()
               if by_key[k]["raw_id"] is not None and seed_raw_counts[str(by_key[k]["raw_id"])] == 1}
    if method == "raw_track":
        for frame in frames:
            raw_counts = Counter(str(p["raw_id"]) for p in frame["players"] if p["raw_id"] is not None)
            for obs in frame["players"]:
                if obs["key"] not in output:
                    if obs["raw_id"] is not None and raw_counts[str(obs["raw_id"])] > 1:
                        record(obs, None, "duplicate_raw_id")
                        continue
                    pid = raw_map.get(str(obs["raw_id"])) if obs["raw_id"] is not None else None
                    record(obs, pid, "same_raw_id" if pid else "new_raw_id")
        return output

    for direction in (1, -1):
        states = {pid: State(by_key[k], seed["frame"], np.zeros(2),
                            [by_key[k]["appearance"]] if by_key[k].get("appearance") is not None else [])
                  for k, pid in labels.items()}
        sequence = sorted((f for f in frames if (f["f"]-seed["frame"])*direction > 0),
                          key=lambda f: f["f"], reverse=direction < 0)
        for frame in sequence:
            observations = frame["players"]
            if not observations:
                continue
            identities = list(states)
            costs = np.full((len(observations), len(identities)), 100.)
            for i, obs in enumerate(observations):
                height = max(1., obs["bb"][3]-obs["bb"][1])
                for j, pid in enumerate(identities):
                    state = states[pid]
                    delta = frame["f"] - state.frame
                    gap = abs(delta)/fps
                    if gap > max_gap_seconds:
                        continue
                    old_height = max(1., state.obs["bb"][3]-state.obs["bb"][1])
                    scale = (height + old_height)/2
                    predicted = centre(state.obs) + state.velocity*np.clip(delta, -.5*fps, .5*fps)
                    distance = float(np.linalg.norm(centre(obs)-predicted)/scale)
                    gate = .65 + min(gap, 2.)*.7
                    if distance > gate or abs(np.log(height/old_height)) > .85:
                        continue
                    cost = .8*distance/gate + .2*min(abs(np.log(height/old_height)), 1.)
                    if method == "motion_appearance":
                        distances = [appearance_distance(obs.get("appearance"), feat) for feat in state.gallery]
                        distances = [d for d in distances if d is not None]
                        if distances:
                            appearance_cost = min(distances)
                            if appearance_cost > .65 and obs["occlusion"] < .3:
                                continue
                            # Under overlap, a crop can mostly contain the other player.
                            weight = .55 if obs["occlusion"] < .3 else .15
                            cost = (1-weight)*cost + weight*appearance_cost
                    costs[i, j] = cost
            # Each observation has its own unknown slot; no forced roster fit.
            augmented = np.concatenate([costs, np.full((len(observations), len(observations)), .72)], axis=1)
            rows, columns = linear_sum_assignment(augmented)
            for i, j in zip(rows, columns):
                obs = observations[i]
                if j >= len(identities):
                    record(obs, None, "unmatched_or_gap")
                    continue
                cost = float(costs[i, j])
                row_alternatives = np.delete(costs[i], j)
                column_alternatives = np.delete(costs[:, j], i)
                alternative = min(np.min(row_alternatives, initial=100.), np.min(column_alternatives, initial=100.))
                if alternative-cost < ambiguity_margin:
                    record(obs, None, "ambiguous", cost)
                    continue
                pid = identities[j]
                record(obs, pid, "associated", cost)
                state = states[pid]
                delta = frame["f"]-state.frame
                velocity = (centre(obs)-centre(state.obs))/delta
                state.velocity = .5*state.velocity + .5*velocity
                state.obs, state.frame = obs, frame["f"]
                if obs["occlusion"] < .3 and obs.get("appearance") is not None:
                    # Preserve seed plus recent clean views, never overwrite seed.
                    state.gallery.append(obs["appearance"])
                    if len(state.gallery) > 8:
                        state.gallery = state.gallery[:1] + state.gallery[-7:]
    return output


def score(predictions: dict, checks: list[dict], seed_keys: set[str]) -> dict:
    """Sparse human checks only, excluded from seed/association. No fake IDF1."""
    checked = [(p, c["player_id"]) for c in checks
               if c.get("verified") is True and c.get("label_source") == "human"
               and c.get("player_id") is not None and c["key"] not in seed_keys
               and (p := predictions.get(c["key"])) is not None]
    assigned = sum(p["player_id"] is not None for p, _ in checked)
    correct = sum(p["player_id"] == label for p, label in checked)
    return {"checked": len(checked), "assigned": assigned, "correct": correct,
            "accuracy_including_unknown": correct/len(checked) if checked else None,
            "accuracy_when_assigned": correct/assigned if assigned else None,
            "coverage_on_checks": assigned/len(checked) if checked else None}


def compare(observations: dict, seed: dict, checks: list[dict] = (), **kwargs) -> dict:
    methods = {}
    for method in METHODS:
        predictions = run_method(observations["frames"], seed, observations["fps"], method, **kwargs)
        counts = Counter(p["reason"] for p in predictions.values())
        assigned = sum(p["player_id"] is not None for p in predictions.values())
        methods[method] = {"predictions": predictions, "diagnostics": {
            "observations": len(predictions), "assigned": assigned,
            "coverage": assigned/max(len(predictions), 1), "reasons": dict(counts),
        }, "human_checks": score(predictions, checks, set(seed["assignments"]))}
    return {"schema": SCHEMA, "source": observations["source"], "seed": seed,
            "parameters": {"max_gap_seconds": kwargs.get("max_gap_seconds", 3.),
                           "ambiguity_margin": kwargs.get("ambiguity_margin", .08)},
            "methods": methods}
