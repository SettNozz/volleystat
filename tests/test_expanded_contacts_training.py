"""Label provenance, held-out source boundaries and feature-only evidence."""
from copy import deepcopy

import numpy as np

from scripts.evaluation.train_expanded_contacts import (
    OLD, NewlineCompatibleFeatures, eligible_event, event_metrics, normalise_event, pooled_event_features, raw_event_indices, reference_side_conflicts, refine_appearance, soft_features, training_channel_mask, training_sources, vote_labels,
)


def test_training_sources_excludes_entire_test_video_for_both_controls():
    groups = [*OLD, "new_a", "new_b"]
    for heldout in groups:
        for expanded in (False, True):
            sources = training_sources(groups, heldout, expanded)
            assert heldout not in sources
            assert sources == (set(groups)-{heldout} if expanded else OLD-{heldout})


def test_author_label_does_not_verify_model_event_and_explicit_type_can_survive_unsure_author():
    c = {"id": "new:f10", "video_id": "new", "type": "pass", "event_label_source": "model"}
    data = {"decisions": {c["id"]: {"decision": "player", "verified": True}}, "event_reviews": {}}
    assert not eligible_event(data, c)
    c["event_label_source"] = "human"
    assert not eligible_event(data, c)
    data["event_reviews"][c["id"]] = {"type": "pass"}
    assert eligible_event(data, c)
    data["decisions"][c["id"]]["decision"] = "unsure"
    assert eligible_event(data, c)
    data["decisions"][c["id"]]["decision"] = "not_contact"
    assert not eligible_event(data, c)


def test_pooled_features_are_per_example_and_do_not_mix_sources():
    x = np.zeros((2, 120, 18), np.float32); x[1] = 100
    a = pooled_event_features(x)
    assert a.shape == (2, 360)
    np.testing.assert_array_equal(a[:1], pooled_event_features(x[:1]))


def test_inactive_event_channels_cannot_activate_from_heldout_values():
    x = np.zeros((3, 120, 18), np.float32)
    x[:2, :, 0] = np.linspace(0, 1, 120)
    x[2, :, 10:] = 1000.
    train = np.array([True, True, False]); active = training_channel_mask(x, train)
    assert active[0] and not active[10:].any()
    mean = x[train].mean(axis=(0, 1)); std = x[train].std(axis=(0, 1)); std[std < 1e-6] = 1
    norm = normalise_event(x, mean, std, active)
    assert not norm[2, :, 10:].any()
    np.testing.assert_array_equal(norm[:2], normalise_event(x[:2], mean, std, active))


def test_soft_identity_features_do_not_use_author_or_raw_roster_numbers():
    p = {"bb": [0, 0, 20, 100], "kp": [[10, 20, .9] for _ in range(17)], "team": "team_b", "id": 91, "player_id": "wrong"}
    frames = {10: {"poses": [p], "ball": [10, 20]}}
    contact = {"frame": 10, "author": {"player_id": "P1"}}
    vectors = {"10:0": np.array([1., 0.]), "a": np.array([1., 0.]), "b": np.array([0., 1.])}
    before = deepcopy(frames)
    a = soft_features(frames, contact, ["P1", "P2"], vectors, {"P1": ["a"], "P2": ["b"]}, 200)
    contact["author"]["player_id"] = "P2"
    b = soft_features(frames, contact, ["P1", "P2"], vectors, {"P1": ["a"], "P2": ["b"]}, 200)
    assert a == b and frames == before
    assert a["P1"]["soft_app3"] < a["P2"]["soft_app3"]
    frames[10]["ball"] = None
    missing = soft_features(frames, contact, ["P1", "P2"], vectors, {"P1": ["a"], "P2": ["b"]}, 200)
    assert missing["P1"]["soft_d3_0.25"] == 9.


def test_event_metrics_exposes_missing_class_support_and_confusion():
    m = event_metrics(np.array([0, 1, 2, 3]), np.array([0, 2, 2, 3]))
    assert m["accuracy"] == .75
    assert m["per_class"]["reception"]["recall"] == 0
    assert m["confusion_matrix"][1][2] == 1


def test_raw_six_class_targets_preserve_block_and_do_not_guess_plain_pass_subtype():
    rows = [{"raw_types": v} for v in (["attack"], ["block"], ["pass"], ["set", "dig"], ["dig"], ["other"])]
    assert raw_event_indices(rows) == [0, 1, 4]


def test_all_sides_can_recover_human_roster_candidate_despite_weak_wrong_pose_side():
    pose = {"team": "a", "bb": [0, 0, 20, 100], "kp": [[10, 20, .9] for _ in range(17)]}
    frames = {10: {"poses": [pose], "ball": [10, 20]}}
    vectors = {"10:0": np.array([1., 0.]), "reference": np.array([1., 0.])}
    contact = {"frame": 10}; gallery = {"P8": ["reference"]}
    gated = soft_features(frames, contact, ["P8"], vectors, gallery, 200)
    both = soft_features(frames, contact, ["P8"], vectors, gallery, 200, all_sides=True)
    assert gated["P8"]["soft_app3"] == 1
    assert both["P8"]["soft_app3"] == 0
    assert pose["team"] == "a" and "player_id" not in pose


def test_source_router_uses_only_unambiguous_identity_setup_side_conflicts():
    roster = {"players": [{"player_id": "P8", "references": [{"frame": 10, "bbox": [0, 0, 20, 100]}]}]}
    far = {"team": "a", "bb": [0, 0, 20, 100]}
    assert reference_side_conflicts(roster, {10: {"poses": [far]}})[0]["person"] == "P8"
    assert not reference_side_conflicts(roster, {10: {"poses": [{**far, "team": "b"}]}})
    assert not reference_side_conflicts(roster, {10: {"poses": [far, {**far, "team": "b"}]}})
    assert not reference_side_conflicts(roster, {})


def test_fixed_vote_has_no_learned_weights_and_hard_hgb_breaks_ties():
    assert vote_labels(["P4", "P2", "P2"]) == "P2"
    assert vote_labels(["P4", "P2", "P1"]) == "P4"


def test_clean_track_refinement_keeps_frozen_embeddings_and_uses_no_targets():
    pose = {"id": 1, "team": "b", "bb": [0, 0, 20, 100], "kp": [[10, 20, .9] for _ in range(17)]}
    ann = {"fps": 30, "frames": [{"f": f, "poses": [deepcopy(pose)]} for f in range(4)]}
    vectors = {"0:0": np.array([1., 0.]), "3:0": np.array([.8, .6]), "a": np.array([1., 0.]), "b": np.array([0., 1.])}
    before = {k: v.copy() for k, v in vectors.items()}
    refined, gallery, audit = refine_appearance(ann, {0, 3}, vectors, {"P1": ["a"], "P2": ["b"]})
    assert audit["pooled_observations"] == 2 and audit["labels_used"] is False
    assert refined["0:0"][1] > 0
    for key in before: np.testing.assert_array_equal(vectors[key], before[key])


def test_legacy_cache_reuse_requires_only_newline_difference_and_identical_runtime(tmp_path, monkeypatch):
    from scripts.evaluation import train_expanded_contacts as module
    from src.models.inference_cache import ModelOutputCache, ObservationFeatures, file_digest
    current_code = tmp_path/"src/models/player_reid_encoder.py"
    current_code.parent.mkdir(parents=True); current_code.write_bytes(b"def encode():\r\n    pass\r\n")
    original = tmp_path/"legacy"; original.mkdir()
    (original/"encoder_source.py").write_bytes(b"def encode():\n    pass\n")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source = tmp_path/"video.webm"; source.write_bytes(b"source")
    cache = ModelOutputCache(source, tmp_path/"cache")
    spec = {"backend": file_digest(current_code), "weights_sha256": "same_weights", "runtime": {"torch": "same"}, "settings": {"flip": True}}
    oldspec = {**spec, "backend": file_digest(original/"encoder_source.py")}
    old = ObservationFeatures(cache, oldspec, 100, 100)
    old.put(10, [0, 0, 10, 10], np.array([1., 2.], np.float32))
    cache.put("imports", "original", {"source": str(original)})
    current = ObservationFeatures(cache, spec, 100, 100)
    reader = NewlineCompatibleFeatures(current)
    np.testing.assert_array_equal(reader.get(10, [0, 0, 10, 10]), [1, 2])
    assert current.get(10, [0, 0, 10, 10]) is None  # never copied into a different stage
    changed = ObservationFeatures(cache, {**spec, "runtime": {"torch": "different"}}, 100, 100)
    assert NewlineCompatibleFeatures(changed).get(10, [0, 0, 10, 10]) is None
    cache.db.close()
