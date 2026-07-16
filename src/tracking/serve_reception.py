"""
Serve-centric rally derivation (rules over FSM events).

The classifier struggles with serve/reception, but the rally STRUCTURE gives
them away almost for free:

    serve      = first contact of a rally (FSM emits it from SERVE_READY)
    reception  = next contact by the receiving team (FSM emits it in SERVING)
    outcome    = what happened right after:
        - next contact by the receiving team (pass/dig/attack)  -> "success"
        - rally ends shortly after the reception                -> "failed"
          (shanked ball: out of bounds / on the floor)
        - no reception at all before rally end                  -> "no_reception"
          (ace or serve fault — ball never reached a receiver's hands)

These derived labels serve two purposes:
    1. Analytics: per-rally serve/reception summary in *_stats.json
    2. Training data: rule-derived labels for serve/reception contacts are
       far more reliable than the sequence model — export them as pre-labels
       (see scripts/export_contact_labels.py) and let the model focus on the
       genuinely hard classes (pass vs attack vs block).

Event dicts are the ones GameAnalyzer builds from RallyEvent:
    {"frame", "time", "type", "game_state", "rally_phase", "team", "contact_seq"}
"""

from typing import Dict, List, Optional

# Rally must continue at least this long after a reception for it to count as
# controlled; if it dies sooner (and nobody else touched), the ball was shanked.
RECEPTION_FAIL_WINDOW_SEC = 2.0


def derive_serve_reception(
    events: List[Dict],
    rally_end_frame: int,
    fps: float = 30.0,
) -> Optional[Dict]:
    """
    Derive the serve/reception summary for one rally.

    Args:
        events:          rally's events, chronological (GameAnalyzer format)
        rally_end_frame: frame at which the rally ended
        fps:             for time windows

    Returns dict (or None if the rally has no serve):
        {
          "serve":     {"frame", "team"},
          "reception": {"frame", "team", "outcome"} | None,
          "outcome":   "success" | "failed" | "no_reception",
        }
    """
    serve = next((e for e in events if e["type"] == "serve"), None)
    if serve is None:
        return None

    reception = next(
        (e for e in events
         if e["type"] == "reception" and e["frame"] > serve["frame"]),
        None,
    )

    if reception is None:
        return {
            "serve": {"frame": serve["frame"], "team": serve["team"]},
            "reception": None,
            "outcome": "no_reception",
        }

    # Contact by the receiving team after the reception = controlled ball
    followup = next(
        (e for e in events
         if e["frame"] > reception["frame"]
         and e["team"] == reception["team"]
         and e["type"] in ("pass", "dig", "attack")),
        None,
    )

    if followup is not None:
        outcome = "success"
    else:
        # Nobody on the receiving side touched again. If the rally died right
        # away, the reception failed; if the ball crossed and play went on
        # (opponent contacts), the reception effectively went over — count it
        # as success-by-crossing only when the rally continued.
        opponent_touched = any(
            e["frame"] > reception["frame"] and e["team"] != reception["team"]
            for e in events
        )
        died_quickly = (
            rally_end_frame - reception["frame"] <= RECEPTION_FAIL_WINDOW_SEC * fps
        )
        outcome = "success" if opponent_touched and not died_quickly else "failed"

    return {
        "serve": {"frame": serve["frame"], "team": serve["team"]},
        "reception": {
            "frame": reception["frame"],
            "team": reception["team"],
            "outcome": outcome,
        },
        "outcome": outcome,
    }


def annotate_rallies(rallies: List[Dict], fps: float = 30.0) -> Dict[str, int]:
    """
    Enrich rally dicts in place with "serve_reception" and return counters.

    Returns e.g. {"serves": 9, "receptions": 8, "success": 6, "failed": 2,
                  "no_reception": 1}
    """
    counts = {"serves": 0, "receptions": 0,
              "success": 0, "failed": 0, "no_reception": 0}
    for rally in rallies:
        sr = derive_serve_reception(
            rally.get("events", []), rally.get("end_frame", 0), fps,
        )
        if sr is None:
            continue
        rally["serve_reception"] = sr
        counts["serves"] += 1
        if sr["reception"] is not None:
            counts["receptions"] += 1
        counts[sr["outcome"]] += 1
    return counts


def derive_contact_labels(rallies: List[Dict]) -> List[Dict]:
    """
    Rule-derived labels for every serve/reception contact — reliable
    pre-labels for the event-classifier dataset.

    Returns: [{"frame", "label", "team", "source", "confidence"}]
    ``confidence`` reflects rule reliability, not a model score.
    """
    labels: List[Dict] = []
    for rally in rallies:
        sr = rally.get("serve_reception")
        if sr is None:
            continue
        labels.append({
            "frame": sr["serve"]["frame"],
            "label": "serve",
            "team": sr["serve"]["team"],
            "source": "fsm_rule",
            "confidence": 0.95,   # first rally contact is a serve by structure
        })
        if sr["reception"] is not None:
            labels.append({
                "frame": sr["reception"]["frame"],
                "label": "reception",
                "team": sr["reception"]["team"],
                "source": "fsm_rule",
                "confidence": 0.9,
                "outcome": sr["reception"]["outcome"],
            })
    return labels
