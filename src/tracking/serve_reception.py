"""Serve-centric derivation of confirmed service-attempt lifecycles.

The FSM provides contact events; these helpers turn them into stable analytics
and training metadata.  A playable rally requires either:

    serve candidate -> net crossing -> opponent reception
    -> another controlled contact by the receiving team

or a reception that returns across the net as an overpass/free ball.
"""

from typing import Dict, List, Optional


RECEPTION_FAIL_WINDOW_SEC = 2.0  # legacy import compatibility
_CONTROLLED_FOLLOWUP_TYPES = {"pass", "dig", "attack"}


def derive_rally_lifecycle(
    events: List[Dict],
    rally_end_frame: int,
) -> Dict:
    """Derive the canonical lifecycle for one service attempt.

    Optional lifecycle snapshots from the FSM provide exact crossing/end
    frames. Legacy result files remain supported through event inference.
    """
    del rally_end_frame  # kept for symmetry with derive_serve_reception
    ordered = sorted(events, key=lambda event: event.get("frame", -1))
    serve = next((e for e in ordered if e.get("type") == "serve"), None)
    serve_fault = next(
        (e for e in ordered if e.get("type") == "serve_fault"), None
    )
    attempt = serve or serve_fault

    if attempt is None:
        return {
            "status": "unconfirmed",
            "is_playable": False,
            "serve_confirmed": False,
            "start_frame": None,
            "serve_frame": None,
            "net_cross_frame": None,
            "reception_frame": None,
            "followup_frame": None,
            "reception_return_frame": None,
            "serving_team": None,
            "receiving_team": None,
            "end_reason": None,
            "serve_outcome": "unknown",
            "reception_quality": "not_applicable",
        }

    reception = next(
        (
            e for e in ordered
            if e.get("type") == "reception"
            and e.get("frame", -1) > attempt.get("frame", -1)
        ),
        None,
    )
    followup = None
    opponent_followup = None
    if reception is not None:
        followup = next(
            (
                e for e in ordered
                if e.get("frame", -1) > reception.get("frame", -1)
                and e.get("team") == reception.get("team")
                and e.get("type") in _CONTROLLED_FOLLOWUP_TYPES
            ),
            None,
        )
        opponent_followup = next(
            (
                e for e in ordered
                if e.get("frame", -1) > reception.get("frame", -1)
                and e.get("team") != reception.get("team")
                and e.get("type") in _CONTROLLED_FOLLOWUP_TYPES | {"block"}
            ),
            None,
        )

    if serve_fault is not None:
        status = "serve_fault"
    elif reception is None:
        status = "ace_no_reception"
    elif followup is None and opponent_followup is None:
        status = "reception_failed"
    else:
        status = "playable_rally"

    event_lifecycle = next(
        (
            e.get("lifecycle") for e in reversed(ordered)
            if isinstance(e.get("lifecycle"), dict)
        ),
        {},
    )
    serving_team = attempt.get("team")
    receiving_team = (
        reception.get("team") if reception is not None else
        ("team_b" if serving_team == "team_a" else "team_a")
        if serving_team in ("team_a", "team_b") else None
    )

    reception_return_frame = event_lifecycle.get("reception_return_frame")
    if reception_return_frame is None and opponent_followup is not None:
        reception_return_frame = opponent_followup.get("frame")
    if serve_fault is not None:
        serve_outcome = "fault"
    elif reception is None:
        serve_outcome = "ace"
    else:
        serve_outcome = "in_play"
    if reception is None:
        reception_quality = "no_touch_ace"
    elif event_lifecycle.get("reception_quality"):
        reception_quality = event_lifecycle["reception_quality"]
    elif reception_return_frame is not None or opponent_followup is not None:
        reception_quality = "overpass"
    elif followup is not None:
        reception_quality = "controlled"
    else:
        reception_quality = "error"

    return {
        "status": status,
        "is_playable": status == "playable_rally",
        "serve_confirmed": serve is not None,
        "start_frame": attempt.get("frame") if status == "playable_rally" else None,
        "serve_frame": attempt.get("frame"),
        "net_cross_frame": event_lifecycle.get("net_cross_frame"),
        "reception_frame": reception.get("frame") if reception else None,
        "followup_frame": followup.get("frame") if followup else None,
        "reception_return_frame": reception_return_frame,
        "serving_team": serving_team,
        "receiving_team": receiving_team,
        "end_reason": event_lifecycle.get("end_reason"),
        "serve_outcome": serve_outcome,
        "reception_quality": reception_quality,
    }


def _merge_lifecycle(derived: Dict, existing: object) -> Dict:
    """Prefer an exact terminal FSM snapshot over event-only inference."""
    if not isinstance(existing, dict):
        return derived
    merged = dict(derived)
    for key, value in existing.items():
        if value is not None or key in ("status", "is_playable", "serve_confirmed"):
            merged[key] = value
    return merged


def derive_serve_reception(
    events: List[Dict],
    rally_end_frame: int,
    fps: float = 30.0,
) -> Optional[Dict]:
    """Derive serve/reception details with the canonical lifecycle outcome."""
    del fps  # retained in the public signature for compatibility
    lifecycle = derive_rally_lifecycle(events, rally_end_frame)
    if lifecycle["serve_frame"] is None:
        return None

    ordered = sorted(events, key=lambda event: event.get("frame", -1))
    reception = next(
        (
            e for e in ordered
            if e.get("type") == "reception"
            and e.get("frame", -1) > lifecycle["serve_frame"]
        ),
        None,
    )
    return {
        "serve": {
            "frame": lifecycle["serve_frame"],
            "team": lifecycle["serving_team"],
            "confirmed": lifecycle["serve_confirmed"],
        },
        "reception": (
            {
                "frame": reception["frame"],
                "team": reception["team"],
                "outcome": lifecycle["status"],
                "quality": lifecycle["reception_quality"],
            }
            if reception is not None else None
        ),
        "outcome": lifecycle["status"],
        "serve_outcome": lifecycle["serve_outcome"],
        "reception_quality": lifecycle["reception_quality"],
        "legacy_outcome": (
            "success" if lifecycle["status"] == "playable_rally"
            else "failed" if lifecycle["status"] == "reception_failed"
            else "no_reception"
        ),
    }


def annotate_rallies(rallies: List[Dict], fps: float = 30.0) -> Dict[str, int]:
    """Enrich attempts in place and return canonical plus legacy counters."""
    counts = {
        "service_attempts": 0,
        "confirmed_serves": 0,
        "playable_rallies": 0,
        "aces_no_reception": 0,
        "reception_failed": 0,
        "serve_faults": 0,
        "controlled_receptions": 0,
        "reception_overpasses": 0,
        "reception_errors": 0,
        "unknown_receptions": 0,
        # Legacy aliases retained for existing callers/UI.
        "serves": 0,
        "receptions": 0,
        "success": 0,
        "failed": 0,
        "no_reception": 0,
    }
    for rally in rallies:
        lifecycle = _merge_lifecycle(
            derive_rally_lifecycle(
                rally.get("events", []), rally.get("end_frame", 0),
            ),
            rally.get("lifecycle"),
        )
        rally["lifecycle"] = lifecycle
        sr = derive_serve_reception(
            rally.get("events", []), rally.get("end_frame", 0), fps,
        )
        if sr is None:
            continue

        rally["serve_reception"] = sr
        sr["outcome"] = lifecycle["status"]
        sr["serve_outcome"] = lifecycle["serve_outcome"]
        sr["reception_quality"] = lifecycle["reception_quality"]
        sr["legacy_outcome"] = (
            "success" if lifecycle["status"] == "playable_rally"
            else "failed" if lifecycle["status"] == "reception_failed"
            else "no_reception"
        )
        sr["serve"]["confirmed"] = lifecycle["serve_confirmed"]
        if sr["reception"] is not None:
            sr["reception"]["outcome"] = lifecycle["status"]
            sr["reception"]["quality"] = lifecycle["reception_quality"]
        counts["service_attempts"] += 1
        if lifecycle["serve_confirmed"]:
            counts["confirmed_serves"] += 1
            counts["serves"] += 1
        if sr["reception"] is not None:
            counts["receptions"] += 1

        status = lifecycle["status"]
        if status == "playable_rally":
            counts["playable_rallies"] += 1
            counts["success"] += 1
        elif status == "ace_no_reception":
            counts["aces_no_reception"] += 1
            counts["no_reception"] += 1
        elif status == "reception_failed":
            counts["reception_failed"] += 1
            counts["failed"] += 1
        elif status == "serve_fault":
            counts["serve_faults"] += 1
            counts["no_reception"] += 1

        quality = lifecycle.get("reception_quality")
        if quality == "controlled":
            counts["controlled_receptions"] += 1
        elif quality == "overpass":
            counts["reception_overpasses"] += 1
        elif quality == "error":
            counts["reception_errors"] += 1
        elif sr["reception"] is not None:
            counts["unknown_receptions"] += 1

    return counts


def derive_contact_labels(rallies: List[Dict]) -> List[Dict]:
    """Return rule-derived serve/reception pre-labels for training."""
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
            "confidence": 0.95 if sr["serve"].get("confirmed") else 0.8,
            "outcome": sr["outcome"],
        })
        if sr["reception"] is not None:
            labels.append({
                "frame": sr["reception"]["frame"],
                "label": "reception",
                "team": sr["reception"]["team"],
                "source": "fsm_rule",
                "confidence": 0.9,
                "outcome": sr["reception"]["outcome"],
                "reception_quality": sr["reception"].get("quality", "unknown"),
            })
    return labels
