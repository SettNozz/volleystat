"""Final fit of the contact spotter on every gold video, for production use.

The leave-one-video-out experiment answers "how good is this method". This
answers "give me one model to run on a new video": it trains on all three gold
sources at once and writes a checkpoint that carries the decode constraints and
the normalisation with it.

There is no held-out video here, so there is nothing to select an epoch on.
That is a real limitation, not an oversight: across the experiment's folds and
seeds the best epoch landed anywhere between 3 and 15, so a single number
cannot be read off them. This trains for a fixed budget and averages several
seeds, which is what makes the result insensitive to that choice -- the
ensemble was measured to help, and averaging removes the need to guess one
seed's best epoch. The epoch budget is an engineering decision and is recorded
as such in the checkpoint provenance.

    python scripts/models/train_contact_spotter.py --help

Reviewed queues (``--reviewed``) add videos that carry human-verified contacts
but no phase track. They enter the loss only at their contacts, never as
background, and never feed the decode constraints; see
``action_recognition/rally_spotting/reviewed_contacts.py``.

The resulting checkpoint is loadable by
``action_recognition.rally_spotting.contact_spotter.ContactSpotter`` and by
``analyze_video.py --contact-spotter <path>``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from action_recognition.rally_spotting import constrained_decode as cd
from action_recognition.rally_spotting import dense_contacts as dc
from action_recognition.rally_spotting import reviewed_contacts as rc
from action_recognition.rally_spotting import spotter_dataset as sd
from action_recognition.rally_spotting.contact_spotter import (
    DEFAULT_STRIDE, DEFAULT_WINDOW, SCHEMA, UNBOUNDED, save_checkpoint,
)
from action_recognition.rally_spotting.model import RallySpottingTCN
from action_recognition.rally_spotting.prepare import FEATURE_NAMES

STEMS = ("Khe4P8QcJkg_5_4k", "j0gRhgFCgIs_4_4k", "j3BO9GgUFBI_3_4k")


def log(message: str) -> None:
    print(f"[train-spotter] {message}", flush=True)


def enforce_determinism() -> dict[str, Any]:
    """Make a rerun of the same seeds produce the same checkpoint.

    Without this, two runs of one seed were measured about 0.01 macro F1
    apart, because cuDNN picks algorithms from what the device has free. A
    production checkpoint that cannot be rebuilt is not auditable.
    """
    report: dict[str, Any] = {"requested": True}
    if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        report["cublas_workspace_config"] = "set here"
    else:
        report["cublas_workspace_config"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        report["warning"] = (
            "the CUDA context already existed, so CUBLAS_WORKSPACE_CONFIG may "
            "not have taken effect"
        )
        log("WARNING: " + report["warning"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
        report["use_deterministic_algorithms"] = True
    except Exception as error:
        report["use_deterministic_algorithms"] = False
        report["error"] = str(error)
        log(f"WARNING: deterministic algorithms unavailable: {error}")
    return report


def train_member(
    seed: int, sources: Sequence[dict[str, Any]], mean: np.ndarray,
    deviation: np.ndarray, criterion: nn.Module, device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """One model over every source, for a fixed number of epochs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RallySpottingTCN(
        input_dim=len(FEATURE_NAMES), num_classes=dc.NUM_LABELS,
        hidden_dim=args.hidden_dim, dilations=tuple(args.dilations),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    features = [
        ((source["features"] - mean) / deviation).astype(np.float32)
        for source in sources
    ]
    targets = [source["targets"] for source in sources]
    items = sd.training_windows(
        sources, args.window, args.stride, args.contact_repeat,
    )
    generator = np.random.default_rng(seed)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = generator.permutation(len(items))
        losses = []
        for position in range(0, len(order), args.batch_size):
            batch = [items[i] for i in order[position:position + args.batch_size]]
            x = np.zeros((len(batch), args.window, len(FEATURE_NAMES)), np.float32)
            y = np.full((len(batch), args.window), dc.IGNORE_INDEX, np.int64)
            for row, (index, start) in enumerate(batch):
                end = min(start + args.window, features[index].shape[0])
                x[row, :end - start] = features[index][start:end]
                y[row, :end - start] = targets[index][start:end]
            logits = model(torch.from_numpy(x).to(device))
            loss = criterion(
                logits.reshape(-1, dc.NUM_LABELS),
                torch.from_numpy(y).to(device).reshape(-1),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "loss": mean_loss})
        log(f"  seed {seed} epoch {epoch:03d} loss={mean_loss:.4f}")
    return {
        "seed": int(seed),
        "history": history,
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "config": model.config(),
    }


def load_sources(
    args: argparse.Namespace, stems: Sequence[str],
    reviewed_stems: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Gold sources, then reviewed ones, with the audits that describe them.

    ``reviewed_stems`` limits which videos of the reviewed queues are used;
    ``None`` takes all of them.
    """
    review = dc.reviewed_contacts(args.reviews)
    log(f"review audit: {review['audit']}")
    sources = []
    for stem in stems:
        source = sd.load_source(
            stem, args.annotations_dir, args.gold_dir,
            review["contacts"].get(stem, []), review["masked"].get(stem, []),
            args.human_source, args.contact_radius, args.guard,
        )
        log(
            f"{stem}: {source['frames']} frames, "
            f"{source['verified_contacts']} verified contacts, "
            f"{source['target_audit']['masked_frames']} masked frames, "
            f"{source['pose_feature_frames']} frames with pose features, "
            f"net_y={source['net_y']:.1f}"
        )
        sources.append(source)

    queues = []
    for path in args.reviewed or []:
        queue = rc.load_review(path)
        chosen = [
            stem for stem in queue["videos"]
            if reviewed_stems is None or stem in reviewed_stems
        ]
        queues.append({
            "path": queue["review_path"], "sha256": queue["review_sha256"],
            "stems": chosen, "audit": queue["audit"],
        })
        for stem in chosen:
            source = rc.load_reviewed_source(
                stem, queue, args.contact_radius, args.guard,
            )
            log(
                f"{stem} (reviewed queue): {source['frames']} frames, "
                f"{source['verified_contacts']} human contacts, "
                f"{source['target_audit']['segments']} typed, "
                f"everything else masked"
            )
            sources.append(source)
    if reviewed_stems is not None:
        found = {s["stem"] for s in sources if s.get("sparse")}
        missing = sorted(set(reviewed_stems) - found)
        if missing:
            raise ValueError(f"reviewed stems not in any queue: {missing}")
    return sources, review, queues


def count_constraints(
    sources: Sequence[dict[str, Any]], args: argparse.Namespace,
) -> dict[str, Any]:
    """Decode constraints from the sources that carry a human phase track.

    Counted on the unguarded copies, because the guard ring separates contacts
    from phases and would forbid them all. A reviewed queue has no phase
    track, so it never contributes a transition.
    """
    structures = [s["structure"] for s in sources if s["structure"] is not None]
    if not structures:
        raise ValueError("no source carries a phase track to count constraints on")
    return cd.estimate_constraints(
        structures, duration_quantile=args.duration_quantile,
        duration_slack=args.duration_slack, self_loop_labels=UNBOUNDED,
    )


def train_members(
    sources: Sequence[dict[str, Any]], args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    """Normalise, weight and train every seed; the shipped recipe."""
    mean, deviation = sd.normalisation(sources)
    weights = sd.class_weights(
        [source["targets"] for source in sources], args.max_weight, args.weight_mode,
    )
    log("class weights " + ", ".join(
        f"{name}={weights[index]:.1f}" for index, name in enumerate(dc.LABEL_TYPES)
    ))
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).to(device), ignore_index=dc.IGNORE_INDEX,
    )
    log(
        f"training {len(args.seeds)} member(s) on {len(sources)} source(s) "
        f"for {args.epochs} epochs on {device}"
    )
    members = [
        train_member(seed, sources, mean, deviation, criterion, device, args)
        for seed in args.seeds
    ]
    return members, mean, deviation, weights


def build_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument(
        "--reviews", type=Path,
        default=ROOT / "data/contact_spatial_gold_v1/reviews.json",
    )
    parser.add_argument(
        "--annotations-dir", type=Path,
        default=ROOT / "results/gold_pose_fsm_20261028_v1",
        help="finished analyses of the gold videos; must carry poses",
    )
    parser.add_argument("--gold-dir", type=Path, default=ROOT / "data/full_video_gold")
    parser.add_argument(
        "--human-source", type=Path,
        default=ROOT / "data/full_video_gold/j3BO9GgUFBI_3_4k/annotations_v1.json",
        help="cumulative human project file, used only to mask uncovered touches",
    )
    parser.add_argument("--stems", nargs="+", default=list(STEMS))
    parser.add_argument(
        "--reviewed", type=Path, action="append", metavar="REVIEW_JSON",
        help="a contact_review_v1 queue to add as sparse human contacts; "
             "repeatable",
    )
    parser.add_argument("--contact-radius", type=int, default=2)
    parser.add_argument("--guard", type=int, default=3)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--inference-stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--contact-repeat", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--dilations", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64],
    )
    parser.add_argument(
        "--epochs", type=int, default=8,
        help="fixed budget; no held-out video exists here to select one",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-weight", type=float, default=25.0)
    parser.add_argument("--weight-mode", choices=("max", "median"), default="max")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--transition-weight", type=float, default=1.0)
    parser.add_argument("--duration-quantile", type=float, default=0.0)
    parser.add_argument("--duration-slack", type=float, default=1.5)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--no-deterministic", dest="deterministic", action="store_false",
        help="allow cuDNN autotuning; the checkpoint is then not reproducible",
    )
    parser.set_defaults(deterministic=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.add_argument("--output", type=Path, required=True, metavar="CHECKPOINT")
    args = parser.parse_args(argv)

    if args.output.exists():
        parser.error(f"refusing to overwrite existing {args.output}")
    determinism = (
        enforce_determinism() if args.deterministic else {"requested": False}
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    sources, review, queues = load_sources(args, args.stems)
    if not sources:
        parser.error("no sources loaded")
    if any(source["pose_feature_frames"] == 0 for source in sources):
        log(
            "WARNING: a source carries no pose evidence; its eight pose "
            "channels are zero and the checkpoint will inherit that"
        )

    members, mean, deviation, _ = train_members(sources, args, device)
    constraints = count_constraints(sources, args)
    log(
        f"constraints: {len(constraints['transition_counts'])} observed "
        f"transitions over {constraints['sequences']} sources"
    )

    provenance = {
        "schema": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "trained_on": [
            {
                "stem": source["stem"],
                "frames": source["frames"],
                "verified_contacts": source["verified_contacts"],
                "annotations": source["annotation_path"],
                "annotation_sha256": source["annotation_sha256"],
                "dense_targets_sha256": source["dense_targets_sha256"],
                "pose_feature_frames": source["pose_feature_frames"],
                "net_y": source["net_y"],
                "target_audit": source["target_audit"],
                "label_source": source.get("label_source", "human_full_video"),
                "sparse": bool(source.get("sparse")),
                "review_audit": source.get("review_audit"),
            }
            for source in sources
        ],
        "reviewed_queues": queues,
        "review_path": str(args.reviews.resolve()),
        "review_sha256": review["review_sha256"],
        "review_audit": review["audit"],
        "label_source": "human",
        "seeds": list(args.seeds),
        "epochs": int(args.epochs),
        "epoch_selection": (
            "fixed budget. Every gold video is in the training set, so no "
            "held-out video exists to select an epoch on; across the "
            "experiment's folds and seeds the best epoch ranged 3-15. The "
            "seed ensemble is what makes this choice less decisive."
        ),
        "determinism": determinism,
        "arguments": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "limits": [
            f"Trained on {sum(not s.get('sparse') for s in sources)} fully "
            f"annotated video(s) and {sum(bool(s.get('sparse')) for s in sources)} "
            "reviewed queue(s). Reviewed videos teach contacts only: their "
            "phases and every frame the queue never offered are masked.",
            "Cannot tell warm-up from play: on the one gold video with a real "
            "warm-up the decoder labels it rally_live and emits contacts "
            "there. Exclude non-game time with --match-config.",
            "Does not attribute a contact to a team or a player.",
            "No held-out estimate accompanies this checkpoint. For measured "
            "quality see the leave-one-video-out experiment in "
            "docs/dense_contact_spotting.md and "
            "docs/contact_detection_vs_typing.md.",
        ],
    }

    save_checkpoint(
        args.output,
        members=[member["state_dict"] for member in members],
        config=members[0]["config"],
        mean=mean,
        deviation=deviation,
        constraints=constraints,
        provenance=provenance,
        window=args.window,
        stride=args.inference_stride,
        transition_weight=args.transition_weight,
    )
    log(f"wrote {args.output} with {len(members)} member(s)")
    sidecar = args.output.with_suffix(args.output.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {"provenance": provenance,
             "training_history": [
                 {"seed": member["seed"], "history": member["history"]}
                 for member in members
             ]},
            indent=1, default=str,
        ),
        encoding="utf-8",
    )
    log(f"provenance sidecar {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
