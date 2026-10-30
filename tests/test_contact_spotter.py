"""A production spotter checkpoint must carry everything inference needs.

The decode constraints are counted from the training videos, so a checkpoint
without them cannot be decoded at all on a new video. These tests pin that
contract, the refusal to run on mismatched feature channels, and the fact that
the spotter reports rather than hides what it cannot do.
"""

import numpy as np
import pytest
import torch

from action_recognition.rally_spotting import constrained_decode as cd
from action_recognition.rally_spotting import dense_contacts as dc
from action_recognition.rally_spotting import spotter_dataset as sd
from action_recognition.rally_spotting.contact_spotter import (
    SCHEMA, UNBOUNDED, ContactSpotter, phase_segments, save_checkpoint,
    window_starts,
)
from action_recognition.rally_spotting.model import RallySpottingTCN
from action_recognition.rally_spotting.prepare import (
    FEATURE_NAMES, build_continuous_features,
)


def _constraints() -> dict:
    # A rally that serves, receives, attacks and stops, so the counted grammar
    # permits those transitions.
    phases = {"between": 0, "serve_flight": 1, "rally_live": 2}
    sequence = (
        [phases["between"]] * 20
        + [dc.contact_index("serve")] * 4
        + [phases["serve_flight"]] * 10
        + [dc.contact_index("reception")] * 4
        + [phases["rally_live"]] * 10
        + [dc.contact_index("attack")] * 4
        + [phases["rally_live"]] * 10
        + [phases["between"]] * 20
    )
    return cd.estimate_constraints(
        [sequence], duration_quantile=0.0, duration_slack=1.5,
        self_loop_labels=UNBOUNDED,
    )


def _checkpoint(tmp_path, members=1, feature_names=None):
    # Seeded, so two checkpoints built here hold the same weights and can be
    # compared to each other rather than to a fresh random initialisation.
    torch.manual_seed(0)
    model = RallySpottingTCN(
        input_dim=len(FEATURE_NAMES), num_classes=dc.NUM_LABELS, hidden_dim=8,
        dilations=(1, 2),
    )
    path = tmp_path / "spotter.pt"
    save_checkpoint(
        path,
        members=[model.state_dict() for _ in range(members)],
        config=model.config(),
        mean=np.zeros(len(FEATURE_NAMES)),
        deviation=np.ones(len(FEATURE_NAMES)),
        constraints=_constraints(),
        provenance={"trained_on": ["unit-test"]},
        window=64, stride=32,
    )
    if feature_names is not None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["feature_names"] = feature_names
        torch.save(payload, path)
    return path


def _annotations(frames=200, width=3840, height=2160):
    return {
        "fps": 29.97, "frame_width": width, "frame_height": height,
        "net_y": height / 2.0,
        "frames": [
            {
                "f": index,
                "ball": [width / 2.0, height / 2.0],
                "p": [[100.0, 200.0]],
                "pose_f": [0.0] * 8,
            }
            for index in range(frames)
        ],
    }


def test_window_starts_always_cover_the_last_frame():
    assert window_starts(10, 64, 32) == [0]
    starts = window_starts(200, 64, 32)
    assert starts[0] == 0
    assert starts[-1] == 200 - 64


def test_checkpoint_round_trips_its_constraints(tmp_path):
    path = _checkpoint(tmp_path)
    spotter = ContactSpotter.load(path, device=torch.device("cpu"))
    assert spotter.constraints["num_labels"] == dc.NUM_LABELS
    assert "rally_live->attack" in spotter.constraints["transition_counts"]
    assert spotter.window == 64 and spotter.stride == 32


def test_a_checkpoint_needs_at_least_one_member(tmp_path):
    model = RallySpottingTCN(input_dim=len(FEATURE_NAMES), num_classes=dc.NUM_LABELS)
    with pytest.raises(ValueError, match="at least one"):
        save_checkpoint(
            tmp_path / "empty.pt", members=[], config=model.config(),
            mean=np.zeros(1), deviation=np.ones(1), constraints={}, provenance={},
        )


def test_experiment_checkpoint_is_refused(tmp_path):
    path = tmp_path / "experiment.pt"
    torch.save({"schema": "dense_contact_spotting_v1", "state_dict": {}}, path)
    with pytest.raises(ValueError, match="not production-loadable"):
        ContactSpotter.load(path, device=torch.device("cpu"))


def test_different_feature_channels_are_refused(tmp_path):
    path = _checkpoint(tmp_path, feature_names=["ball_x", "ball_y"])
    with pytest.raises(ValueError, match="different feature channels"):
        ContactSpotter.load(path, device=torch.device("cpu"))


def test_run_returns_events_in_source_frame_numbers(tmp_path):
    spotter = ContactSpotter.load(
        _checkpoint(tmp_path), device=torch.device("cpu"),
    )
    result = spotter.run(_annotations())
    assert result.phase_labels.shape[0] == 200
    assert result.frame_numbers[0] == 0 and result.frame_numbers[-1] == 199
    for event in result.events:
        assert 0 <= event["frame"] <= 199
        assert event["type"] in dc.CONTACT_TYPES
        assert event["core_type"] in dc.CORE_TYPES
        assert event["source"] == "dense_contact_spotter"


def test_diagnostics_say_warm_up_is_not_handled(tmp_path):
    spotter = ContactSpotter.load(
        _checkpoint(tmp_path), device=torch.device("cpu"),
    )
    result = spotter.run(_annotations())
    assert "match-config" in result.diagnostics["warm_up_note"]
    assert result.diagnostics["schema"] == SCHEMA
    assert result.diagnostics["members"] == 1


def test_an_ensemble_averages_its_members(tmp_path):
    _, features = build_continuous_features(_annotations())
    single = ContactSpotter.load(
        _checkpoint(tmp_path), device=torch.device("cpu"),
    ).probabilities(features)

    three = tmp_path / "three"
    three.mkdir()
    ensemble = ContactSpotter.load(
        _checkpoint(three, members=3), device=torch.device("cpu"),
    )
    assert len(ensemble.members) == 3
    # Identical members must average back to one member's distribution, and
    # the average has to stay a distribution for the decoder's logarithms.
    assert np.allclose(ensemble.probabilities(features), single, atol=1e-6)
    assert np.allclose(single.sum(axis=1), 1.0, atol=1e-6)


def test_phase_segments_exclude_contacts():
    labels = np.array(
        [0] * 5 + [dc.contact_index("serve")] * 3 + [1] * 4, dtype=np.int64,
    )
    segments = phase_segments(labels, np.arange(labels.shape[0]))
    assert [segment["phase"] for segment in segments] == ["between", "serve_flight"]
    assert segments[0]["start_frame"] == 0 and segments[0]["end_frame"] == 4
    assert segments[1]["frames"] == 4


def test_class_weights_lift_contacts_against_the_largest_class():
    targets = [np.array([0] * 900 + [dc.contact_index("serve")] * 10)]
    weights = sd.class_weights(targets, maximum=25.0, mode="max")
    assert weights[0] == pytest.approx(1.0)
    assert weights[dc.contact_index("serve")] == pytest.approx(25.0)
    # A label never seen keeps a neutral weight rather than exploding.
    assert weights[dc.contact_index("block")] == pytest.approx(1.0)


def test_normalisation_guards_a_constant_channel():
    sources = [{"features": np.tile(np.array([[1.0, 5.0]]), (10, 1))}]
    mean, deviation = sd.normalisation(sources)
    assert mean.tolist() == [1.0, 5.0]
    # A zero standard deviation would divide every row to infinity.
    assert deviation.tolist() == [1.0, 1.0]


def test_training_windows_repeat_contact_bearing_windows():
    targets = np.zeros(400, dtype=np.int64)
    targets[100:105] = dc.contact_index("attack")
    sources = [{"frames": 400, "targets": targets}]
    items = sd.training_windows(sources, window=100, stride=100, contact_repeat=3)
    starts = [start for _, start in items]
    assert starts.count(100) == 3
    assert starts.count(0) == 1
