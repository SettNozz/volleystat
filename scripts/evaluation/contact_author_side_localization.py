"""Test two contact-author upgrades, separately and together (LOVO).

1. Side: ball-box-size depth cues (per contact and summed per possession)
   added to the near/other-team model. Sizes are normalised by the video's
   own median contact ball size, so no net geometry is needed.
2. Localization: per-candidate cues for occluded contacts: hand distance to
   the contact point reconstructed from the ball path before/after, wrist
   speed, jump height and hands-above-head on the candidate's own continuous
   track segment.

Every arm is refit leave-one-video-out on the same rows with the same budget;
the control refits the unchanged features. Inputs are the stored feature
pickles of ``compare_contact_author_identity_methods.py features`` plus the
saved analyses. No inference, no source edits.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import pickle
from pathlib import Path
import sys
from statistics import median

import numpy as np

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))

from src.tracking.contact_author_person_ranker import chain_posterior, contact_distance, pose_geometry  # noqa: E402
from src.tracking.contact_author_runtime import track_segments  # noqa: E402
from src.tracking.contact_side import CORE_TYPE, _possessions  # noqa: E402

SEEDS = (42, 43, 44)
MISSING = -9.0
SIDE_COLUMNS = ["side_log0", "side_n0", "side_log_pre", "side_log_post", "side_possession", "side_alternated",
                "ball_y_net", "ball_size_ref"]
LOC_COLUMNS = ["loc_traj_d", "loc_traj_d_near", "loc_wrist_speed", "loc_jump", "loc_hands_up", "loc_dt",
               "loc_ball_missing", "loc_traj_sides"]


# ------------------------------------------------------------------ ball

def observed_ball(frames):
    out = {}
    for r in frames:
        if r.get("ball") is None or r.get("ball_interp") or r.get("ball_out_of_frame"):
            continue
        out[r["f"]] = (float(r["ball"][0]), float(r["ball"][1]), float(r["ball_size"]) if r.get("ball_size") else None)
    return out


def median_size(obs, a, b):
    vals = [obs[f][2] for f in range(a, b + 1) if f in obs and obs[f][2]]
    return (float(median(vals)), len(vals)) if vals else (None, 0)


def contact_point(obs, f):
    """Ball position at the contact frame from the observed path around it."""
    if f in obs:
        return np.array(obs[f][:2]), 2
    sides = []
    for frames in (sorted(x for x in range(f - 12, f) if x in obs)[-4:],
                   sorted(x for x in range(f + 1, f + 13) if x in obs)[:4]):
        if len(frames) >= 2:
            t = np.array(frames, float); xy = np.array([obs[x][:2] for x in frames])
            sides.append(np.array([np.polyval(np.polyfit(t, xy[:, k], 1), f) for k in (0, 1)]))
        elif len(frames) == 1 and abs(frames[0] - f) <= 3:
            sides.append(np.array(obs[frames[0]][:2]))
    return (np.mean(sides, axis=0), len(sides)) if sides else (None, 0)


# ------------------------------------------------------------------ side

def side_features(contacts, obs, frame_height, net_y):
    sizes = [median_size(obs, c["frame"] - 5, c["frame"] + 5)[0] for c in contacts]
    known = [s for s in sizes if s]
    ref = float(median(known)) if known else None
    rows = []
    for c, s in zip(contacts, sizes):
        n = median_size(obs, c["frame"] - 5, c["frame"] + 5)[1]
        pre = median_size(obs, c["frame"] - 15, c["frame"] - 6)[0]
        post = median_size(obs, c["frame"] + 6, c["frame"] + 15)[0]
        log = lambda v: float(np.log(v / ref)) if v and ref else MISSING  # noqa: E731
        point, _ = contact_point(obs, c["frame"])
        rows.append({"side_log0": log(s), "side_n0": n, "side_log_pre": log(pre), "side_log_post": log(post),
                     "ball_y_net": float((point[1] - net_y) / frame_height) if point is not None and net_y else MISSING,
                     "ball_size_ref": ref or MISSING})
    # Possession evidence: summed non-attack size evidence, alternation inside a rally.
    order = sorted(range(len(contacts)), key=lambda i: contacts[i]["frame"])
    seq = [{"frame": contacts[i]["frame"], "core": CORE_TYPE.get(contacts[i]["type"], "pass")} for i in order]
    groups = _possessions(seq, 4.0 * 30)
    score = {}
    for g in groups:
        ev = [rows[order[i]]["side_log0"] for i in g if seq[i]["core"] != "attack" and rows[order[i]]["side_log0"] != MISSING]
        if not ev:
            ev = [rows[order[i]]["side_log0"] for i in g if rows[order[i]]["side_log0"] != MISSING]
        for i in g:
            score[order[i]] = float(np.sum(ev)) if ev else 0.
    for gi, g in enumerate(groups):
        alt = 0.
        if gi and seq[g[0]]["core"] != "serve":
            alt = -np.sign(score[order[groups[gi - 1][0]]])
        for i in g:
            rows[order[i]]["side_possession"] = score[order[i]]
            rows[order[i]]["side_alternated"] = float(alt)
    return np.array([[r[k] for k in SIDE_COLUMNS] for r in rows], np.float32)


# ------------------------------------------------------------------ localization

def segment_views(observations, tracks, frame, index):
    return tracks[observations[(frame, index)]["segment"]]


def kp(pose, j):
    k = (pose.get("kp") or [])
    return (k[j][0], k[j][1]) if len(k) > j and k[j][2] >= .3 else None


def localization_features(annotations, contacts, features, observations, tracks, obs):
    by_frame = {r["f"]: r for r in annotations["frames"]}
    out = []
    for i, c in enumerate(contacts):
        f = c["frame"]
        point, sides = contact_point(obs, f)
        missing = float(not any(x in obs for x in range(f - 3, f + 4)))
        rows = []
        for pid in features["options"]:
            actor = features["evidence"][i].get(pid) if pid != "other_team" else None
            if not actor or (actor["frame"], actor["pose_index"]) not in observations:
                rows.append([MISSING, MISSING, MISSING, MISSING, MISSING, MISSING, missing, sides]); continue
            pose = by_frame[actor["frame"]]["poses"][actor["pose_index"]]
            views = segment_views(observations, tracks, actor["frame"], actor["pose_index"])
            near = min(views, key=lambda v: abs(v["frame"] - f))["pose"]
            h = max(pose["bb"][3] - pose["bb"][1], 1.)
            g = pose_geometry(pose, point) if point is not None else None
            gn = pose_geometry(near, point) if point is not None else None
            local = sorted((v for v in views if abs(v["frame"] - f) <= 6), key=lambda v: v["frame"])
            speed = 0.
            for a, b in zip(local, local[1:]):
                for j in (9, 10):
                    pa, pb = kp(a["pose"], j), kp(b["pose"], j)
                    if pa and pb:
                        speed = max(speed, np.hypot(pb[0] - pa[0], pb[1] - pa[1]) / h / max(b["frame"] - a["frame"], 1))
            ankles = lambda v: [kp(v["pose"], j)[1] for j in (15, 16) if kp(v["pose"], j)]  # noqa: E731
            base = [y for v in views if -30 <= v["frame"] - f <= -10 for y in ankles(v)]
            now = [y for v in local for y in ankles(v)]
            jump = (float(median(base)) - min(now)) / h if base and now else MISSING
            nose, wrists = kp(near, 0), [kp(near, j) for j in (9, 10) if kp(near, j)]
            hands_up = float(min(w[1] for w in wrists) < nose[1]) if nose and wrists else MISSING
            rows.append([contact_distance(g) if g else MISSING, contact_distance(gn) if gn else MISSING, speed, jump,
                         hands_up, abs(actor["frame"] - f), missing, sides])
        out.append(np.array(rows, np.float32))
    return out


# ------------------------------------------------------------------ stages

def stage_features(args):
    out = args.output / "extended"; out.mkdir(parents=True, exist_ok=True)
    for path in sorted((args.output / "features").glob("*.pkl")):
        vid = path.stem; target = out / f"{vid}.pkl"
        if target.exists():
            continue
        item = pickle.load(path.open("rb"))
        ann = json.loads((args.benchmark / "pipeline" / vid / f"{vid}_annotations.json").read_bytes())
        obs = observed_ball(ann["frames"])
        observations, tracks = track_segments(ann["frames"], ann["fps"])
        result = {"side": side_features(item["contacts"], obs, ann["frame_height"], ann.get("net_y")), "loc": {}}
        for variant in ("auto_gallery", "human_photos"):
            result["loc"][variant] = localization_features(ann, item["contacts"], item["variants"][variant]["features"],
                                                           observations, tracks, obs)
        pickle.dump(result, target.open("wb"))
        print(f"{vid}: side {result['side'].shape}, observed ball frames {len(obs)}", flush=True)


def load(args):
    items, extra = {}, {}
    for path in sorted((args.output / "features").glob("*.pkl")):
        items[path.stem] = pickle.load(path.open("rb"))
        extra[path.stem] = pickle.load((args.output / "extended" / path.name).open("rb"))
    return items, extra


def truth_of(c):
    return c["author"]["player_id"] if c["author"]["decision"] == "player" else "other_team"


def guard(item, c):
    return all(abs(c["frame"] - f) > 60 for f in item["photo_frames"])


def ranker_rows(item, extra, variant, use_loc):
    payload = item["variants"][variant]["features"]; options = payload["options"]
    inverse = {h: a for a, h in item["mapping"].items()} if variant != "human_photos" else None
    xs, ys = [], []
    for i, (c, x) in enumerate(zip(item["truth"], payload["candidate_features"])):
        if not guard(item, c):
            continue
        t = truth_of(c)
        target = t if inverse is None or t == "other_team" else inverse.get(t)
        if target not in options:
            continue
        if use_loc:
            x = np.concatenate([x, extra["loc"][variant][i]], axis=1)
        xs.append(x); ys.append([int(o == target) for o in options])
    return xs, ys


def fit(items, extra, held, use_loc, use_side):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    X, Y, G, N = [], [], [], []
    for vid, item in items.items():
        if vid == held:
            continue
        for variant in ("human_photos", "auto_gallery"):
            xs, ys = ranker_rows(item, extra[vid], variant, use_loc)
            X.extend(xs); Y.extend(ys)
        gx = item["variants"]["auto_gallery"]["features"]["global_features"]
        if use_side:
            gx = np.concatenate([gx, extra[vid]["side"]], axis=1)
        for i, c in enumerate(item["truth"]):
            if guard(item, c):
                G.append(gx[i]); N.append(int(truth_of(c) != "other_team"))
    X, Y = np.concatenate(X), np.concatenate(Y)
    rankers = [HistGradientBoostingClassifier(max_iter=300, learning_rate=.04, max_leaf_nodes=15, l2_regularization=1.,
               min_samples_leaf=10, early_stopping=False, random_state=s).fit(X, Y) for s in SEEDS]
    teams = [ExtraTreesClassifier(400, min_samples_leaf=2, max_features=.3, random_state=s, n_jobs=4).fit(np.array(G), N)
             for s in SEEDS]
    return rankers, teams


def stage_score(args):
    sys.path.insert(0, str(WORKTREE / "scripts/evaluation"))
    import compare_contact_author_identity_methods as cmp
    scorer = cmp.load_scorer(args.benchmark)
    class A: pass
    a = A(); a.root = args.root; a.experiment = args.experiment
    _, lovo = cmp.load_models(a)
    items, extra = load(args)
    arms = {"control": (False, False), "side": (False, True), "localization": (True, False), "both": (True, True)}
    rows = defaultdict(list); team_rows = defaultdict(list)
    for held, item in items.items():
        for arm, (use_loc, use_side) in arms.items():
            rankers, teams = fit(items, extra, held, use_loc, use_side)
            gx = item["variants"]["auto_gallery"]["features"]["global_features"]
            if use_side:
                gx = np.concatenate([gx, extra[held]["side"]], axis=1)
            emission = np.mean([t.predict_proba(gx)[:, 1] for t in teams], axis=0)
            posterior = chain_posterior(item["contacts"], emission, lovo[held]["transitions"], {held})
            for i, c in enumerate(item["truth"]):
                team_rows[arm].append(int((posterior[i] >= .5) == (truth_of(c) != "other_team")))
            for variant in ("auto_gallery", "human_photos"):
                f = item["variants"][variant]["features"]
                for i, (c, x) in enumerate(zip(item["truth"], f["candidate_features"])):
                    if use_loc:
                        x = np.concatenate([x, extra[held]["loc"][variant][i]], axis=1)
                    s = np.mean([m.predict_proba(x)[:, 1] for m in rankers], axis=0)
                    pid = f["options"][int(np.argmax(s[:-1]))]
                    p = "other_team" if posterior[i] < .5 else pid
                    rows[(variant, arm)].append({"video_id": held, "id": c["id"], "type": c["type"], "truth": truth_of(c),
                        "predicted": cmp.label_for(p, item["mapping"], variant == "human_photos"), "raw": p})
            print(f"{held} {arm}: done", flush=True)
    summary = []
    for (variant, arm), method_rows in rows.items():
        s = scorer.author_scores(method_rows)
        per_video = {v: scorer.author_scores([r for r in method_rows if r["video_id"] == v])["correct_named"]
                     for v in sorted(items)}
        summary.append({"roster": variant, "arm": arm, "correct": s["correct_named"], "named": s["named_targets"],
                        "recall": s["exact_person_recall"], "precision": s["exact_person_precision"],
                        "f1": s["person_f1"], "author_or_other": s["author_or_other_team_accuracy"],
                        "other_team_recall": s["other_team_recall"], "errors": s["named_errors"],
                        "team_accuracy": float(np.mean(team_rows[arm])), "best_naming": cmp.best_naming(method_rows),
                        "per_video": per_video})
    (args.output / "extended_rows").mkdir(exist_ok=True)
    for (variant, arm), method_rows in rows.items():
        cmp.write(args.output / "extended_rows" / f"{variant}__{arm}.json", method_rows)
    cmp.write(args.output / "extended_summary.json", summary)
    for r in summary:
        print(f"{r['roster']:13} {r['arm']:13} {r['correct']}/{r['named']} rec {r['recall']:.3f} prec {r['precision']:.3f} "
              f"F1 {r['f1']:.3f} acc {r['author_or_other']:.3f} team {r['team_accuracy']:.3f} best {r['best_naming']} {r['errors']}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["features", "score"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--benchmark", type=Path, required=True)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    (stage_features if args.stage == "features" else stage_score)(args)


if __name__ == "__main__":
    main()
