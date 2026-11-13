"""Promoted author contract, source-bound setup, replay and default integration."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from src.tracking import contact_author_runtime as runtime


class ProbabilityHead:
    def __init__(self, constant=None): self.constant = constant
    def predict_proba(self, x):
        p = np.full(len(x), self.constant) if self.constant is not None else np.asarray(x)[:, 0]
        return np.stack([1-p, p], axis=1)


@pytest.fixture
def model(tmp_path):
    manifest = {"schema": "contact_author_expanded_candidate_v2", "selected_stage": "authors_all_sides",
        "selected_method": "expanded_soft_hgb", "source_videos": ["training"],
        "feature_views": {"all_sides": {"soft_columns": ["x"], "global_columns": ["g"]}}}
    path = tmp_path/"model.joblib"
    joblib.dump({"manifest": manifest, "heads": {"all_sides/hgb_soft": [ProbabilityHead() for _ in range(3)]},
        "team_models": [ProbabilityHead(.9) for _ in range(3)], "transitions": {}}, path)
    return runtime.ContactAuthorModel(path, expected_sha256=runtime.sha(path))


def pose(x=10, raw=1):
    return {"id": raw, "team": "a", "player_id": "old_wrong_id", "bb": [x, 0, x+20, 100],
        "kp": [[x+10, 30, .9] for _ in range(17)]}


def annotations():
    return {"fps": 30, "frame_width": 200, "frame_height": 200,
        "frames": [{"f": i, "poses": [pose(), pose(80, 2)], "ball": [20, 30]} for i in range(31)]}


def test_checksum_is_checked_before_deserializing(tmp_path, monkeypatch):
    path = tmp_path/"model.joblib"; path.write_bytes(b"not a model")
    monkeypatch.setattr(joblib, "load", lambda *_: pytest.fail("untrusted pickle was opened"))
    with pytest.raises(ValueError, match="checksum"): runtime.ContactAuthorModel(path)


def test_frozen_head_choice_and_feature_order_guard(model):
    features = {"columns": ["x"], "global_columns": ["g"], "global_features": np.ones((1, 1)),
        "candidate_features": [np.array([[.1], [.8], [.4]])], "options": ["P1", "P8", "other_team"], "evidence": [{}]}
    result = model.predict_features(features, [{"frame": 10, "type": "pass", "video_id": "test"}])
    assert result[0]["label"] == "P8"
    # P8 was substituted off at frame 10: the best on-court player gets the contact.
    benched = model.predict_features(features, [{"frame": 10, "type": "pass", "video_id": "test"}],
                                     on_court={"P8": [[0, 5], [20, 40]]})
    assert benched[0]["label"] == "P1" and benched[0]["scores"][1] == pytest.approx(.8)
    features["columns"] = ["different"]
    with pytest.raises(ValueError, match="schema"): model.predict_features(features, [])


def test_features_ignore_author_targets_and_leave_source_poses_untouched():
    ann = annotations(); before = deepcopy(ann)
    roster = {"players": [{"player_id": p, "references": [{"frame": 15, "bbox": pose(x)["bb"]}]}
        for p, x in (("P1", 10), ("P8", 80))]}
    contacts = [{"frame": 15, "type": "attack", "video_id": "test"}]
    _, requests, _ = runtime.crop_requests(ann, contacts, roster)
    vectors = {p["key"]: np.array([1., 0.], np.float32) if p["bb"][0] == 10 else np.array([0., 1.], np.float32)
        for players in requests.values() for p in players}
    first = runtime.build_features(ann, contacts, roster, vectors)
    contacts[0]["author"] = {"player_id": "P8"}
    second = runtime.build_features(ann, contacts, roster, vectors)
    np.testing.assert_array_equal(first["candidate_features"], second["candidate_features"])
    assert ann == before and first["evidence"][0]["P1"]["raw_track_id"] == 1


def test_five_player_setup_keeps_actual_ids_and_substitute_references():
    ann = {"frames": [{"f": 0, "poses": [pose(i*30, i) for i in range(5)]}],
        "player_identity": {"side": "b", "max_on_court": 5, "seed": {"assignments":
            {f"0:{i}": p for i, p in enumerate(("P1", "P2", "P3", "P7", "P8"))}, "label_source": "human", "verified": True}}}
    roster = runtime.photo_roster(ann)
    assert roster["max_on_court"] == 5 and len(roster["players"]) == 5
    assert "P6" not in [p["player_id"] for p in roster["players"]]
    explicit = {**roster, "video_sha256": "wrong"}
    with pytest.raises(ValueError, match="another source"): runtime.photo_roster(ann, explicit, "actual")


def test_cache_only_missing_crop_never_loads_encoder_or_decodes_video(model, monkeypatch):
    from src.models import inference_cache as ic
    ann = annotations(); roster = {"schema": "contact_author_photo_roster_v1", "video_sha256": "source", "side": "b",
        "players": [{"player_id": "P1", "references": [{"frame": 15, "bbox": pose()["bb"]}]}]}
    monkeypatch.setattr(ic, "FrozenOSNet", lambda *_a, **_kw: SimpleNamespace())
    monkeypatch.setattr(ic, "osnet_feature_store", lambda *_a: SimpleNamespace(get=lambda *_a: None))
    monkeypatch.setattr(runtime, "NewlineCompatibleFeatures", lambda s: s)
    cache = SimpleNamespace(source={"sha256": "source"}, require=False)
    with pytest.raises(ic.CacheMiss, match="missing OSNet"):
        model.attribute(ann, [{"frame": 15, "type": "attack"}], roster=roster, cache=cache,
            require_cache=True, images=lambda *_: pytest.fail("video was decoded"))


def test_wrong_source_is_rejected_before_cache_or_image_access(model):
    ann = annotations(); ann["model_cache"] = {"source": {"kind": "video", "sha256": "original"}}
    with pytest.raises(ValueError, match="different sources"):
        model.attribute(ann, [{"frame": 15, "type": "attack"}], cache=SimpleNamespace(source={"sha256": "other"}))


def test_no_supported_contacts_produces_an_explicit_empty_report(model):
    ann = annotations(); roster = {"schema": "contact_author_photo_roster_v1", "video_sha256": "source", "side": "b",
        "players": [{"player_id": "P1", "references": [{"frame": 15, "bbox": pose()["bb"]}]}]}
    report = model.attribute(ann, [{"frame": 15, "type": "not_a_contact"}], roster=roster,
        cache=SimpleNamespace(source={"sha256": "source"}))
    assert not report["contacts"] and report["attribution"]["status"] == "no_contacts"


def test_pipeline_preserves_learned_report_in_paired_exports(tmp_path, monkeypatch):
    from src.pipeline.game_analyzer import GameAnalyzer, GameStats
    from src.tracking.player_contacts import build_player_contacts
    analyzer = GameAnalyzer.__new__(GameAnalyzer); analyzer.enable_contact_author = True
    analyzer.model_provenance = {}; analyzer.calibration = None; analyzer.net_height_m = 2.34
    stats = GameStats(video_name="test", fps=30); stats.frame_annotations = annotations()["frames"]
    stats.all_events = [{"frame": 15, "type": "attack"}]
    predicted = build_player_contacts(stats.all_events, stats.frame_annotations, 30)
    predicted.update(method=runtime.METHOD, attribution={"status": "completed", "model_resolution": {"sha256": "sha"}})
    monkeypatch.setattr(runtime, "attribute_contacts", lambda *_a, **kw: deepcopy(predicted) if kw["enabled"] else pytest.fail("default disabled"))
    analyzer._apply_contact_authors(stats, "test.webm", 200, 200)
    analyzer._save_and_print(stats, tmp_path, "test")
    for suffix in ("annotations", "stats"):
        data = json.loads((tmp_path/f"test_{suffix}.json").read_bytes())
        assert data["player_contacts"] == predicted
    assert stats.models["contact_author"]["sha256"] == "sha"
