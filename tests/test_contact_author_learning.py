from copy import deepcopy
import json

import numpy as np
import pytest

from src.tracking.contact_author_features import (
    contact_report, extract_events, feature_fingerprint, motion,
)
from scripts.evaluation.run_contact_author_learning import feature_indices, model, write


def fixture():
    frames = []
    for f in range(15):
        poses = []
        for team, x, pid, raw in (("a", 200, None, 1), ("b", 100, "P1", 2)):
            points = [[x+20, 60, .9] for _ in range(17)]
            points[7], points[9] = [x, 90, .9], [x+20, 100, .9]
            points[8], points[10] = [x+30, 90, .9], [x+40, 100, .9]
            poses.append({"team": team, "bb": [x-10, 30, x+60, 150], "kp": points,
                          "id": raw, "player_id": pid,
                          "identity": {"reason": "associated", "appearance_distance": .1}})
        frames.append({"f": f, "ball": [115+f, 101], "poses": poses})
    events = [{"frame": 7, "type": "pass", "source": "dense_contact_spotter"}]
    return events, frames


def test_features_do_not_consume_human_xy_ids_or_absolute_time():
    events, frames = fixture()
    x, names, _, _ = extract_events(events, frames, 30, (1000, 500))
    before = deepcopy(frames)
    changed = deepcopy(frames)
    for ann in changed:
        ann["f"] += 1000
        for pose in ann["poses"]:
            pose["id"] += 100
            if pose["player_id"]: pose["player_id"] = "P999"
    other = [{**events[0], "frame": 1007, "xy": [999999, 999999], "ball": [999999, 999999]}]
    y, fields, _, _ = extract_events(other, changed, 30, (1000, 500))
    np.testing.assert_allclose(x, y, atol=1e-6)
    assert names == fields and frames == before
    assert x.shape[1] == 366 and np.isfinite(x).all()
    assert not any("player_id" in name or "video_id" in name or "raw_id" in name for name in names)


def test_missing_features_have_availability_and_no_ball_motion_bridge():
    events, frames = fixture()
    for ann in frames:
        for pose in ann["poses"]:
            pose["kp"][9][2] = pose["kp"][10][2] = 0
    x, names, _, _ = extract_events(events, frames, 30, (1000, 500))
    assert x[0, names.index("a_actor0_wrist__available")] == 0
    assert x[0, names.index("b_actor0_wrist__available")] == 0
    frames[7]["ball_out_of_frame"] = True
    assert motion({f["f"]: f for f in frames}, 7, 120, 30) == [None]*8+[0., 0.]
    _, _, actors, _ = extract_events(events, frames, 30, (1000, 500))
    assert actors == [[]]


def test_numeric_feature_contract_is_stable_and_layout_type_ablations_are_exact():
    events, frames = fixture()
    _, names, _, _ = extract_events(events, frames, 30, (1000, 500))
    assert len(feature_fingerprint()) == 64
    geometry = feature_indices(names, "geometry")
    typed = feature_indices(names, "typed")
    layout = feature_indices(names, "layout")
    assert set(geometry) < set(typed) < set(layout)
    assert all(not names[i].startswith(("type_", "layout_")) for i in geometry)


def test_author_report_preserves_events_and_does_not_claim_person_supervision():
    events, frames = fixture()
    _, _, actors, indices = extract_events(events, frames, 30, (1000, 500))
    before = deepcopy(events)
    report = contact_report(events, actors, indices, np.asarray([.99]), 30, {"side": "b"}, "learned_test")
    assert report["contacts"][0]["player_id"] == "P1"
    assert report["contacts"][0]["verified"] is False
    assert report["contacts"][0]["score_calibrated"] is False
    assert events == before
    # NumPy bool values must not leave a partially written review bundle.
    json.dumps(report, allow_nan=False)
    unknown = contact_report(events, actors, indices, [.5], 30, {"side": "b"}, "learned_test")
    assert unknown["contacts"][0]["player_id"] is None
    empty = contact_report(events, [[]], [0], [.01], 30, {"side": "b"}, "learned_test")
    assert empty["contacts"][0]["status"] == "unresolved"


def test_models_disable_random_internal_holdout_and_fit_scaler_on_training_only():
    for family in ("logistic", "extra_trees", "hist_gradient"):
        pipeline = model(family)
        x = np.asarray([[0., 0.], [1., 1.], [0., 1.], [1., 0.]])
        pipeline.fit(x, [0, 1, 0, 1])
        np.testing.assert_allclose(pipeline[0].mean_, [.5, .5])
        assert pipeline.predict_proba([[100., 100.]]).shape == (1, 2)
        if family == "hist_gradient": assert pipeline[-1].early_stopping is False


def test_failed_json_serialization_does_not_create_partial_file(tmp_path):
    path = tmp_path/"bad.json"
    with pytest.raises(ValueError): write(path, {"value": float("nan")})
    assert not path.exists()
