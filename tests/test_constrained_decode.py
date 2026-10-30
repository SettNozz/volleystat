"""Constrained decoding must enforce counted structure, not invent it."""

import numpy as np
import pytest

from action_recognition.rally_spotting import constrained_decode as cd
from action_recognition.rally_spotting.dense_contacts import (
    IGNORE_INDEX, NUM_LABELS, contact_index,
)

BETWEEN, SERVE_FLIGHT, RALLY = 0, 1, 2
SERVE = contact_index("serve")
RECEPTION = contact_index("reception")
ATTACK = contact_index("attack")


def _rally_sequence(repeats=3):
    """between -> serve -> serve_flight -> reception -> rally_live -> attack ..."""
    sequence = []
    for _ in range(repeats):
        sequence += [BETWEEN] * 20 + [SERVE] * 3 + [SERVE_FLIGHT] * 10
        sequence += [RECEPTION] * 3 + [RALLY] * 15 + [ATTACK] * 3
    return sequence + [BETWEEN] * 20


def _log_probabilities(labels, confidence=0.9):
    frames = len(labels)
    probabilities = np.full((frames, NUM_LABELS), (1 - confidence) / (NUM_LABELS - 1))
    probabilities[np.arange(frames), labels] = confidence
    return np.log(probabilities)


def test_segments_split_on_masked_frames():
    labels = [BETWEEN, BETWEEN, IGNORE_INDEX, BETWEEN]
    assert cd.segments_of(labels) == [(BETWEEN, 0, 1), (BETWEEN, 3, 3)]


def test_masked_gap_is_not_a_transition():
    labels = [BETWEEN] * 5 + [IGNORE_INDEX] * 3 + [RALLY] * 5
    constraints = cd.estimate_constraints([labels])
    assert "between->rally_live" not in constraints["transition_counts"]


def test_transition_confidence_is_counted_not_assumed():
    labels = _rally_sequence(repeats=2)
    constraints = cd.estimate_constraints([labels])
    counts = constraints["transition_counts"]
    assert counts["between->serve"] == 2
    assert counts["serve->serve_flight"] == 2
    assert counts["attack->between"] == 2
    assert "serve->attack" not in counts
    log_transition = constraints["log_transition"]
    assert log_transition[BETWEEN, SERVE] == pytest.approx(0.0)
    assert log_transition[SERVE, ATTACK] == cd.NEG_INF


def test_duration_bounds_are_frames_not_fractions():
    constraints = cd.estimate_constraints([_rally_sequence(repeats=3)])
    assert constraints["duration_min"][SERVE] == 3
    assert constraints["duration_max"][SERVE] == 3
    assert constraints["duration_min"][RALLY] == 15


def test_unbounded_labels_drop_their_upper_duration_bound():
    constraints = cd.estimate_constraints(
        [_rally_sequence()], self_loop_labels=[BETWEEN],
    )
    assert constraints["duration_max"][BETWEEN] == np.iinfo(np.int64).max
    assert constraints["duration_max"][RALLY] == 15


def test_decoding_recovers_a_clean_sequence():
    labels = _rally_sequence(repeats=3)
    constraints = cd.estimate_constraints([labels], self_loop_labels=[BETWEEN])
    decoded = cd.constrained_viterbi(_log_probabilities(labels), constraints)
    assert decoded.tolist() == labels


def test_decoding_removes_an_illegal_spike():
    labels = _rally_sequence(repeats=3)
    constraints = cd.estimate_constraints([labels], self_loop_labels=[BETWEEN])
    probabilities = np.exp(_log_probabilities(labels))
    # A confident serve in the middle of a rally: legal for a peak picker,
    # forbidden by the counted grammar.
    spike = 20 + 3 + 10 + 3 + 5
    probabilities[spike] = 0.01
    probabilities[spike, SERVE] = 0.95
    decoded = cd.constrained_viterbi(np.log(probabilities), constraints)
    assert decoded[spike] != SERVE
    assert cd.argmax_events(probabilities / probabilities.sum(1, keepdims=True))
    assert all(
        event["type"] != "serve" or event["frame"] != spike
        for event in cd.events_from_labels(decoded, probabilities)
    )


def test_video_end_truncates_the_last_segment_by_default():
    labels = _rally_sequence(repeats=2)
    constraints = cd.estimate_constraints([labels])
    truncated = labels[:-18]  # ends mid-``between``, under its minimum of 20
    decoded = cd.constrained_viterbi(_log_probabilities(truncated), constraints)
    assert decoded[-1] == BETWEEN
    with pytest.raises(cd.InfeasibleConstraints):
        cd.constrained_viterbi(
            _log_probabilities(truncated), constraints,
            enforce_terminal_minimum=True,
        )


def test_events_are_timed_at_the_segment_peak():
    labels = np.array([RALLY] * 10 + [ATTACK] * 5 + [RALLY] * 10)
    probabilities = np.zeros((25, NUM_LABELS))
    probabilities[:, RALLY] = 0.9
    probabilities[10:15, ATTACK] = [0.3, 0.4, 0.95, 0.5, 0.2]
    events = cd.events_from_labels(labels, probabilities)
    assert events == [{
        "frame": 12, "type": "attack", "start_frame": 10, "end_frame": 14,
        "confidence": 0.95,
    }]


def test_events_fall_back_to_the_segment_centre():
    labels = np.array([RALLY] * 5 + [SERVE] * 4 + [RALLY] * 5)
    events = cd.events_from_labels(labels)
    assert events[0]["frame"] == 6 and events[0]["confidence"] is None


def test_unseen_transitions_are_reported_not_hidden():
    train = _rally_sequence(repeats=2)
    constraints = cd.estimate_constraints([train])
    held_out = [BETWEEN] * 20 + [SERVE] * 3 + [ATTACK] * 3 + [BETWEEN] * 20
    missing = cd.unseen_transitions(constraints, [held_out])
    assert missing == {"serve->attack": 1}


def test_infeasible_constraints_raise_rather_than_guess():
    labels = [BETWEEN] * 10 + [SERVE] * 3 + [BETWEEN] * 10
    constraints = cd.estimate_constraints([labels])
    # Only ``between`` may start, and only a 3-frame serve may follow it.
    constraints = dict(constraints)
    constraints["valid_start"] = [SERVE]
    constraints["valid_end"] = [RALLY]
    with pytest.raises(cd.InfeasibleConstraints):
        cd.constrained_viterbi(_log_probabilities(labels), constraints)


def test_argmax_baseline_needs_a_contact_above_the_phases():
    probabilities = np.zeros((10, NUM_LABELS))
    probabilities[:, RALLY] = 0.9
    probabilities[5, ATTACK] = 0.6  # below the phase mass, so not an event
    assert cd.argmax_events(probabilities, threshold=0.5) == []
    probabilities[5] = 0.0
    probabilities[5, ATTACK] = 0.95
    assert [e["frame"] for e in cd.argmax_events(probabilities)] == [5]
