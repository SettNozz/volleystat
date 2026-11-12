"""Results-page statistics preparation (lifecycle counters).

Moved verbatim from viewer.py, which re-exports these names.
"""


_LIFECYCLE_COUNTERS = (
    "service_attempts",
    "confirmed_serves",
    "playable_rallies",
    "aces_no_reception",
    "reception_failed",
    "serve_faults",
    # Spotter-built rallies: a serve the decode missed but its serve
    # probability recovered, and serves that no contact followed.
    "inferred_serves",
    "serves_not_played",
)


def _prepare_viewer_stats(stats: dict) -> dict:
    """Add a small, stable presentation contract without reclassifying events.

    New analysis results provide canonical rally lifecycle metadata.  Legacy
    results predate that schema, so the viewer keeps their original rally list
    and total instead of guessing service outcomes from noisy contact events.
    """
    prepared = dict(stats)
    prepared["rallies"] = [item for item in stats.get("rallies") or [] if not item.get("excluded_from_match_stats")]
    prepared["service_attempts"] = [item for item in stats.get("service_attempts") or [] if not item.get("excluded_from_match_stats")]
    prepared.setdefault("all_events", [])
    prepared["active_event_count"] = sum(not event.get("review_excluded") and not event.get("excluded_from_match_stats")
                                         for event in prepared["all_events"])

    lifecycle = stats.get("rally_lifecycle")
    lifecycle_available = isinstance(lifecycle, dict)
    if lifecycle_available:
        # A counter the producer did not measure stays None: the spotter
        # cannot tell an ace from a fault, so it must not read as zero aces.
        counters = {
            key: int(lifecycle.get(key) or 0) if key in lifecycle else None
            for key in _LIFECYCLE_COUNTERS
        }
        counters["playable_rallies"] = counters["playable_rallies"] or 0
        prepared["rallies"] = [
            rally for rally in prepared["rallies"]
            if isinstance(rally, dict)
            and isinstance(rally.get("lifecycle"), dict)
            and rally["lifecycle"].get("status") == "playable_rally"
            and rally["lifecycle"].get("is_playable") is True
        ]
        # The canonical lifecycle counter is the user-facing rally total.
        prepared["total_rallies"] = counters["playable_rallies"]
    else:
        legacy_total = int(
            stats.get("total_rallies", len(prepared["rallies"])) or 0
        )
        counters = {
            "service_attempts": None,
            "confirmed_serves": None,
            "playable_rallies": legacy_total,
            "aces_no_reception": None,
            "reception_failed": None,
            "serve_faults": None,
            "inferred_serves": None,
            "serves_not_played": None,
        }
        prepared.setdefault("total_rallies", legacy_total)

    prepared["viewer_lifecycle"] = {
        "available": lifecycle_available,
        **counters,
    }
    return prepared
