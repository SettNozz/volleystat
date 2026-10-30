"""Splitting a spotting run into detection and typing must not move the score.

The point of ``score_contact_detection.py`` is that the two halves multiply
back to the end-to-end number, so the pairing it uses has to be the same
pairing the run's own metrics used, and the typing half has to be conditioned
on located contacts only.
"""

import numpy as np
import pytest

from action_recognition.rally_spotting import dense_contacts as dc
from action_recognition.rally_spotting.metrics import match_events
from scripts.evaluation.score_contact_detection import (
    detection_scores,
    detector_proxy,
    nms_frames,
    non_game_mask,
    pair_events,
    typing_scores,
)


def test_pairing_agrees_with_the_counting_matcher():
    predicted = [10, 40, 41, 100]
    truth = [12, 42, 300]
    pairs = pair_events(predicted, truth, 5)
    true_positive, false_positive, false_negative = match_events(
        predicted, truth, 5,
    )
    assert len(pairs) == true_positive
    assert len(predicted) - len(pairs) == false_positive
    assert len(truth) - len(pairs) == false_negative


def test_pairing_takes_the_earliest_compatible_target():
    # A nearest-neighbour rule would pair 14 with 16 and strand 10.
    pairs = pair_events([14], [10, 16], 5)
    assert pairs == [(14, 10)]


def test_nms_collapses_one_touch_and_keeps_separate_ones():
    assert nms_frames([100, 102, 104, 200], 5) == [100, 200]
    assert nms_frames([100, 106], 5) == [100, 106]


def test_detection_scores_every_tolerance():
    scores = detection_scores([102, 400], [100])
    assert scores["at_5"]["tp"] == 1
    assert scores["at_5"]["fp"] == 1
    assert scores["at_2"]["tp"] == 1
    assert scores["at_1"]["tp"] == 0
    assert set(scores) == {f"at_{t}" for t in (1, 2, 4, 5, 10, 15)}


def test_typing_is_conditioned_on_located_contacts():
    truth = [
        {"frame": 100, "type": "serve", "core_type": "serve"},
        {"frame": 500, "type": "attack", "core_type": "attack"},
    ]
    # Only the first contact is located; the second prediction is far away.
    predicted = [
        {"frame": 101, "type": "serve", "confidence": 0.9},
        {"frame": 900, "type": "dig", "confidence": 0.9},
    ]
    result = typing_scores(predicted, truth, 5)
    assert result["located_contacts"] == 1
    assert result["core_accuracy"] == pytest.approx(1.0)
    assert result["core_confusion"] == {"serve->serve": 1}


def test_typing_counts_a_wrong_type_as_a_confusion():
    truth = [{"frame": 100, "type": "dig", "core_type": "pass"}]
    predicted = [{"frame": 100, "type": "attack", "confidence": 0.9}]
    result = typing_scores(predicted, truth, 5)
    assert result["located_contacts"] == 1
    assert result["core_accuracy"] == pytest.approx(0.0)
    assert result["core_confusion"] == {"pass->attack": 1}
    assert result["raw_confusion"] == {"dig->attack": 1}


def test_two_predictions_on_one_frame_keep_the_confident_one():
    truth = [{"frame": 100, "type": "set", "core_type": "pass"}]
    predicted = [
        {"frame": 100, "type": "attack", "confidence": 0.2},
        {"frame": 100, "type": "set", "confidence": 0.8},
    ]
    result = typing_scores(predicted, truth, 5)
    assert result["same_frame_predictions_dropped"] == 1
    assert result["located_contacts"] == 1
    assert result["core_accuracy"] == pytest.approx(1.0)


def test_detector_proxy_reads_the_summed_contact_channels():
    frames = 200
    probabilities = np.zeros((frames, dc.NUM_LABELS), dtype=np.float64)
    probabilities[:, 0] = 1.0
    for frame in (50, 120):
        probabilities[frame, 0] = 0.0
        probabilities[frame, dc.contact_index("attack")] = 0.9
    result = detector_proxy(probabilities, [50, 121], radius=5, tolerance=5)
    assert result["scores"]["at_5"]["tp"] == 2
    assert result["events"] == 2
    assert any(row["threshold"] == pytest.approx(0.5) for row in result["sweep"])


def test_non_game_mask_comes_from_the_human_phase_channel(tmp_path):
    phases = np.zeros(20, dtype=np.int64)
    phases[:8] = dc.EXPORTED_PHASES.index("non_game")
    phases[8:] = dc.EXPORTED_PHASES.index("rally_live")
    directory = tmp_path / "stem" / "derived_7d09206_v2"
    directory.mkdir(parents=True)
    np.savez(
        directory / "dense_targets.npz",
        phase_targets=phases,
        frame_numbers=np.arange(20),
    )
    mask = non_game_mask(tmp_path, "stem", 20)
    assert mask[:8].all()
    assert not mask[8:].any()


def test_non_game_mask_tolerates_a_shorter_export(tmp_path):
    phases = np.full(10, dc.EXPORTED_PHASES.index("non_game"), dtype=np.int64)
    directory = tmp_path / "stem" / "derived_7d09206_v2"
    directory.mkdir(parents=True)
    np.savez(
        directory / "dense_targets.npz",
        phase_targets=phases,
        frame_numbers=np.arange(10),
    )
    mask = non_game_mask(tmp_path, "stem", 14)
    assert mask[:10].all()
    assert not mask[10:].any()


def test_fsm_payload_survives_a_missing_core_type():
    # FSM events carry a bare collapsed type and no ``core_type``.
    truth = [{"frame": 100, "type": "set", "core_type": "pass"}]
    predicted = [{"frame": 102, "type": "pass"}]
    result = typing_scores(predicted, truth, 5)
    assert result["core_accuracy"] == pytest.approx(1.0)
    # ``pass`` is not a raw contact type, so the raw view records a confusion.
    assert result["raw_confusion"] == {"set->pass": 1}
