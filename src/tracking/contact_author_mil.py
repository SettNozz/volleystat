"""Weak team-supervised actor ranking. No person labels are manufactured."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from src.tracking.contact_author_features import ACTOR_FIELDS, _temporal_metrics, geometry

SCHEMA = "contact_author_team_mil_v1"


def pack_instances(events, frames, fps, global_features, global_names, candidates, indices, typed=True):
    by_frame = {ann["f"]: ann for ann in frames}
    context_columns = [i for i, name in enumerate(global_names) if not name.startswith(("a_", "b_", "layout_"))
                       and (typed or not name.startswith("type_"))]
    names = [name for field in ACTOR_FIELDS for name in (field, field+"__available")]
    names += [global_names[i] for i in context_columns]
    x = np.zeros((len(candidates), 16, len(names)), dtype=np.float32)
    team = np.full((len(candidates), 16), -1, dtype=np.int64)
    metadata = []
    for row, (actors, event_index) in enumerate(zip(candidates, indices)):
        # Keep at most eight distinct temporal actors per side, not a raw
        # fixed detector index. Identity values are metadata, never inputs.
        chosen = [actor for s in ("a", "b") for actor in [a for a in actors if a["side"] == s][:8]]
        metadata.append(deepcopy(chosen))
        for column, actor in enumerate(chosen):
            ann = by_frame[actor["frame"]]
            pose = ann["poses"][actor["pose_index"]]
            heights = [p["bb"][3]-p["bb"][1] for p in ann["poses"] if len(p.get("bb") or []) == 4 and p["bb"][3] > p["bb"][1]]
            scale = float(np.median(heights)) if heights else 1.
            g = geometry(pose, ann["ball"], scale)
            vector = g["values"]+[(actor["frame"]-int(events[event_index]["frame"]))/fps]
            vector += _temporal_metrics(by_frame, actor["frame"], pose, g, scale, fps)
            values = [v for value in vector for v in
                      (float(np.clip(value, -5, 5)) if value is not None else 0., float(value is not None))]
            values += list(global_features[row, context_columns])
            x[row, column] = values
            team[row, column] = int(actor["side"] == "b")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite MIL features")
    return x, team, metadata, names


class ActorRanker(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(input_dim, 32), nn.Tanh(), nn.Dropout(.10), nn.Linear(32, 1))

    def forward(self, x, teams):
        actor = self.scorer(x).squeeze(-1)
        pools = []
        for side in (0, 1):
            # Max-like pooling allows one latent contact actor to explain a
            # team's positive bag; no chosen player is treated as human truth.
            masked = (actor/.5).masked_fill(teams != side, -10000.)
            pools.append(torch.logsumexp(masked, dim=1)*.5)
        pooled = torch.stack(pools, dim=1)
        pooled = torch.where((teams >= 0).any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled))
        return pooled, actor


def fit_ranker(x, teams, y, seed, epochs=140):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    present = teams >= 0
    values = x[present]
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-5] = 1.
    normalized = (x-mean)/scale
    useful = (teams == 0).any(axis=1) & (teams == 1).any(axis=1)
    if useful.sum() < 4 or set(y[useful]) != {0, 1}:
        raise ValueError("MIL training needs both labelled sides with observed competing actors")
    data = torch.as_tensor(normalized[useful], dtype=torch.float32, device="cpu")
    groups = torch.as_tensor(teams[useful], dtype=torch.long, device="cpu")
    target = torch.as_tensor(y[useful], dtype=torch.long, device="cpu")
    network = ActorRanker(x.shape[-1]).cpu()
    optimizer = torch.optim.AdamW(network.parameters(), lr=.005, weight_decay=.05)
    counts = np.bincount(y[useful], minlength=2)
    weights = torch.as_tensor(useful.sum()/(2*counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    network.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits, _ = network(data, groups)
        loss = loss_fn(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.)
        optimizer.step()
    network.eval()
    return {"network": network, "mean": mean, "scale": scale,
            "training_usable": int(useful.sum()), "training_total": len(y), "seed": seed, "epochs": epochs}


def predict_rankers(members, x, teams):
    side_scores, shares = [], []
    for member in members:
        normalized = torch.as_tensor((x-member["mean"])/member["scale"], dtype=torch.float32)
        with torch.inference_mode():
            pooled, actor = member["network"](normalized, torch.as_tensor(teams, dtype=torch.long))
            side_scores.append(torch.softmax(pooled, dim=1)[:, 1].numpy())
            conditional = torch.zeros_like(actor)
            for side in (0, 1):
                masked = actor.masked_fill(torch.as_tensor(teams != side), -10000.)
                distribution = torch.softmax(masked, dim=1)
                conditional += torch.where(torch.as_tensor(teams == side), distribution, 0.)
            shares.append(conditional.numpy())
    return np.mean(side_scores, axis=0), np.mean(shares, axis=0)


def ranked_actors(metadata, shares):
    result = []
    for row, actors in enumerate(metadata):
        ranked = []
        for column, actor in enumerate(actors):
            actor = deepcopy(actor)
            actor["geometric_score"] = actor["score"]
            actor["actor_share"] = float(shares[row, column])
            actor["score"] = -float(shares[row, column])
            ranked.append(actor)
        result.append(sorted(ranked, key=lambda a: a["score"]))
    return result
