"""The registry has to be right about *which* weights, and safe when it is down.

Every test here runs against a throw-away sqlite registry, so it exercises the
same MLflow code path as the local server without needing one to be running.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.models import model_registry as mr

mlflow = pytest.importorskip("mlflow", reason="mlflow is optional at import time")


@pytest.fixture()
def registry(tmp_path, monkeypatch):
    """A private registry plus a private artifact cache."""
    uri = "sqlite:///" + str(tmp_path / "registry.db").replace("\\", "/")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(mr, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.chdir(tmp_path)  # keep stray mlruns/ out of the repo
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return uri


@pytest.fixture()
def weights(tmp_path):
    path = tmp_path / "best.pt"
    path.write_bytes(b"weights-v1" * 512)
    return path


def config_for(weights: Path, **overrides):
    entry = {
        "registry_name": "test_ball_detector",
        "alias": "production",
        "artifact": weights.name,
        "fallback": str(weights),
        "imgsz": 1280,
    }
    entry.update(overrides)
    return {"ball_detector": entry}


def register(name: str, entry: dict, path: Path, alias: str = "production", **kw):
    from scripts.models.register_models import register_one
    return register_one(name, entry, path, alias, kw.get("note"), kw.get("force", False), kw.get("companions"))


# ------------------------------------------------------------------ resolving


def test_unregistered_model_falls_back_and_says_so(registry, weights):
    resolved = mr.resolve_model("ball_detector", config=config_for(weights))
    assert resolved.source == "fallback"
    assert resolved.path == weights
    assert "no version with alias" in (resolved.warning or "")
    # The result file must carry the warning, not just the log.
    assert "warning" in resolved.as_dict()


def test_registered_model_is_served_from_the_registry(registry, weights):
    config = config_for(weights)
    register("ball_detector", config["ball_detector"], weights)

    resolved = mr.resolve_model("ball_detector", config=config)
    assert resolved.source == "registry"
    assert resolved.version == "1"
    assert resolved.path.read_bytes() == weights.read_bytes()
    assert resolved.path != weights  # a copy, not the original file


def test_second_resolution_uses_the_cache(registry, weights):
    config = config_for(weights)
    register("ball_detector", config["ball_detector"], weights)
    first = mr.resolve_model("ball_detector", config=config)
    weights.unlink()  # the local file is gone; the registry copy is not

    second = mr.resolve_model("ball_detector", config=config)
    assert second.path == first.path
    assert second.source == "registry"


def test_explicit_path_bypasses_the_registry(registry, weights, tmp_path):
    config = config_for(weights)
    register("ball_detector", config["ball_detector"], weights)
    other = tmp_path / "experiment.pt"
    other.write_bytes(b"something else")

    resolved = mr.resolve_model("ball_detector", override=str(other), config=config)
    assert resolved.source == "override"
    assert resolved.path == other


def test_unreachable_server_prefers_the_cache_over_the_fallback(
    registry, weights, monkeypatch
):
    config = config_for(weights)
    register("ball_detector", config["ball_detector"], weights)
    cached = mr.resolve_model("ball_detector", config=config)

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:59999")
    resolved = mr.resolve_model("ball_detector", config=config)
    assert resolved.source == "cache"
    assert resolved.path == cached.path
    assert "unreachable" in (resolved.warning or "")


def test_registry_can_be_switched_off(registry, weights):
    config = config_for(weights)
    register("ball_detector", config["ball_detector"], weights)
    resolved = mr.resolve_model("ball_detector", config=config, use_registry=False)
    assert resolved.source == "fallback"


def test_unknown_name_is_a_programming_error(registry):
    with pytest.raises(KeyError):
        mr.resolve_model("no_such_model", config={})


# --------------------------------------------------------------- registering


def test_registering_the_same_bytes_twice_reuses_the_version(registry, weights):
    config = config_for(weights)
    first = register("ball_detector", config["ball_detector"], weights)
    second = register("ball_detector", config["ball_detector"], weights)

    assert first["version"] == second["version"]
    assert "already registered" in second["status"]


def test_new_bytes_create_a_version_and_move_the_alias(registry, weights):
    config = config_for(weights)
    register("ball_detector", config["ball_detector"], weights)
    weights.write_bytes(b"weights-v2" * 512)
    second = register("ball_detector", config["ball_detector"], weights)

    assert int(second["version"]) == 2
    resolved = mr.resolve_model("ball_detector", config=config)
    assert resolved.version == "2"
    assert resolved.path.read_bytes() == weights.read_bytes()


def test_version_carries_provenance_tags(registry, weights):
    from mlflow.tracking import MlflowClient

    config = config_for(weights)
    row = register("ball_detector", config["ball_detector"], weights, note="gt window 0.72")
    client = MlflowClient(tracking_uri=registry, registry_uri=registry)
    version = client.get_model_version("test_ball_detector", row["version"])

    assert version.tags["sha256"] == mr.sha256_of(weights)
    assert version.tags["note"] == "gt window 0.72"
    assert json.loads(version.tags["imgsz"]) == 1280


# ------------------------------------------------------------------- config


def test_shipped_config_covers_every_model_the_pipeline_asks_for():
    """A name used in code but missing from the config raises at load time."""
    config = mr.load_config()
    for name in ("ball_detector", "ball_detector_tiled", "pose_estimator",
                 "net_sam", "event_sequence", "player_reid"):
        assert name in config, f"{name} missing from configs/model_registry.json"
        assert config[name].get("fallback"), f"{name} has no fallback path"


def test_home_relative_fallback_is_expanded(registry, weights):
    config = config_for(weights, fallback="~/nowhere/sam.pth")
    resolved = mr.resolve_model("ball_detector", config=config)
    assert "~" not in str(resolved.path)


def test_stopped_local_server_is_detected_without_mlflow_retries(monkeypatch):
    import socket
    import time
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()  # nothing listens on this port now
    uri = f"http://127.0.0.1:{port}"
    monkeypatch.setattr(mr, "_unreachable", {})
    started = time.monotonic()
    with pytest.raises(ConnectionError, match="not listening"):
        mr._require_reachable(uri)
    assert time.monotonic() - started < 2.0
    assert f"127.0.0.1:{port}" in mr._unreachable
    # Remembered: the next logical model in the same analysis does not probe again.
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("probed twice"))
    with pytest.raises(ConnectionError):
        mr._require_reachable(uri)
    mr._require_reachable("sqlite:///unused.db")  # file stores are never probed


# ------------------------------------------------------------------ pins, backup, restore


def _bundle(tmp_path):
    """A main file plus two companions, one of them in a sibling folder."""
    main = tmp_path / "run" / "final" / "model.pt"
    main.parent.mkdir(parents=True)
    main.write_bytes(b"final-weights" * 100)
    sidecar = main.parent / "model.pt.json"
    sidecar.write_text('{"members": 3}', encoding="utf-8")
    fold = tmp_path / "run" / "fold_a" / "model.pt"
    fold.parent.mkdir()
    fold.write_bytes(b"fold-weights" * 100)
    entry = {"registry_name": "test_bundle", "alias": "production", "artifact": "model.pt",
             "fallback": str(main), "sha256": mr.sha256_of(main),
             "files": {"model.pt.json": mr.sha256_of(sidecar), "../fold_a/model.pt": mr.sha256_of(fold)}}
    return entry, main, sidecar, fold


def test_shipped_config_pins_every_required_model():
    for name, entry in mr.load_config().items():
        if entry.get("required", True):
            assert len(entry.get("sha256") or "") == 64, f"{name} has no sha256 pin"
    assert "contact_spotter" in mr.load_config()


def test_fallback_records_its_sha_and_flags_other_bytes(registry, weights):
    config = config_for(weights, sha256=mr.sha256_of(weights))
    resolved = mr.resolve_model("ball_detector", config=config)
    assert resolved.sha256 == mr.sha256_of(weights) and "sha256_mismatch" not in resolved.extra
    weights.write_bytes(b"retrained")
    resolved = mr.resolve_model("ball_detector", config=config)
    assert resolved.extra.get("sha256_mismatch") is True
    assert "differs from the pinned" in resolved.warning
    assert resolved.as_dict()["sha256_mismatch"] is True


def test_missing_fallback_points_at_the_restore_command(registry, tmp_path):
    config = config_for(tmp_path / "gone.pt", sha256="0" * 64)
    resolved = mr.resolve_model("ball_detector", config=config)
    assert resolved.sha256 is None and "verify_models.py --restore" in resolved.warning


def test_check_files_reports_each_state(tmp_path):
    entry, main, sidecar, fold = _bundle(tmp_path)
    assert {r["name"]: r["status"] for r in mr.check_files(entry)} == {
        "model.pt": "ok", "model.pt.json": "ok", "../fold_a/model.pt": "ok"}
    sidecar.write_text("changed", encoding="utf-8")
    fold.unlink()
    states = {r["name"]: r["status"] for r in mr.check_files(entry)}
    assert states == {"model.pt": "ok", "model.pt.json": "mismatch", "../fold_a/model.pt": "missing"}
    assert mr.check_files(dict(entry, sha256=None))[0]["status"] == "unpinned"


def test_register_backs_up_companions_and_restore_brings_them_back(registry, tmp_path, monkeypatch):
    from scripts.models import verify_models as vm
    from mlflow.tracking import MlflowClient
    entry, main, sidecar, fold = _bundle(tmp_path)
    companions = [(f["name"], f["path"], f["sha256"]) for f in mr.check_files(entry) if f["role"] == "companion"]
    register("bundle", entry, main, companions=companions)
    client = MlflowClient(tracking_uri=registry, registry_uri=registry)
    info = (client, registry, False)
    assert vm.verify({"bundle": entry}, client_info=info) == 0

    originals = {p: p.read_bytes() for p in (main, sidecar, fold)}
    main.unlink(); fold.unlink()
    assert vm.verify({"bundle": entry}, client_info=info) == 1
    assert vm.verify({"bundle": entry}, restore=True, client_info=info) == 0
    assert all(p.read_bytes() == data for p, data in originals.items())


def test_restore_never_overwrites_and_reports_other_bytes(registry, tmp_path):
    from scripts.models import verify_models as vm
    from mlflow.tracking import MlflowClient
    entry, main, sidecar, fold = _bundle(tmp_path)
    register("bundle", entry, main, companions=[])
    client = MlflowClient(tracking_uri=registry, registry_uri=registry)
    main.write_bytes(b"someone else's weights")
    assert vm.verify({"bundle": entry}, restore=True, client_info=(client, registry, False)) == 1
    assert main.read_bytes() == b"someone else's weights"


def test_unbacked_pinned_model_is_a_problem_even_without_a_registry(tmp_path):
    from scripts.models import verify_models as vm
    entry, *_ = _bundle(tmp_path)
    assert vm.verify({"bundle": entry}, client_info=(None, None, False)) == 1
    assert vm.verify({"opt": dict(entry, required=False)}, client_info=(None, None, False)) == 0


def test_local_artifact_store_is_read_without_a_server(tmp_path):
    from scripts.models import verify_models as vm
    store = tmp_path / "mlartifacts" / "1" / "run" / "artifacts" / "m"
    store.mkdir(parents=True)
    (store / "w.pt").write_bytes(b"x")
    out = tmp_path / "out"; out.mkdir()
    got = vm.fetch("mlflow-artifacts:/1/run/artifacts/m/w.pt", out, None, False, tmp_path / "mlartifacts")
    assert got.read_bytes() == b"x"


def test_companion_artifact_names_are_unique_and_safe():
    from scripts.models.register_models import companion_artifact
    a = companion_artifact("../fold_a/model.pt")
    b = companion_artifact("../fold_b/model.pt")
    assert a != b and ".." not in a and a.endswith("/model.pt")


def test_entry_without_registry_copy_is_not_a_backup_problem(tmp_path):
    from scripts.models import verify_models as vm
    entry, *_ = _bundle(tmp_path)
    assert vm.verify({"big": dict(entry, backup=False)}, client_info=(None, None, False)) == 0


def test_registry_flags_weights_other_than_the_pin(registry, weights):
    # The alias can point at other bytes than the config pins; imgsz still comes
    # from the config, so the result must say the two may not belong together.
    config = config_for(weights, sha256="0" * 64)
    register("ball_detector", config["ball_detector"], weights)

    resolved = mr.resolve_model("ball_detector", config=config)
    assert resolved.source == "registry"
    assert resolved.extra.get("sha256_mismatch") is True
    assert "differs from the pinned" in (resolved.warning or "")


def test_cache_is_per_registered_model_not_per_logical_name(registry, weights, tmp_path):
    # Moving a logical name to another registered model restarts its version
    # numbers at 1; the old model's cached v1 must not be served for it.
    old = config_for(weights)
    register("ball_detector", old["ball_detector"], weights)
    assert mr.resolve_model("ball_detector", config=old).path.read_bytes() == weights.read_bytes()

    newer = tmp_path / "new" / "best.pt"
    newer.parent.mkdir()
    newer.write_bytes(b"weights-1920" * 512)
    new = config_for(newer, registry_name="test_ball_detector_1920")
    register("ball_detector", new["ball_detector"], newer)

    resolved = mr.resolve_model("ball_detector", config=new)
    assert resolved.version == "1"
    assert resolved.path.read_bytes() == newer.read_bytes()
    assert "sha256_mismatch" not in resolved.extra


def test_shipped_ball_detector_runs_at_the_size_it_was_trained_for():
    entry = mr.load_config()["ball_detector"]
    assert entry["imgsz"] == 1920
    assert entry["registry_name"] == "volleystat_ball_detector_1920"


@pytest.mark.parametrize("override, expected", [(None, 1920), (1280, 1280)])
def test_ball_imgsz_comes_from_the_registry_unless_given(tmp_path, override, expected):
    from src.pipeline.game_analyzer import GameAnalyzer

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"old-1280-weights")
    analyzer = GameAnalyzer.__new__(GameAnalyzer)
    analyzer.use_model_registry = False
    analyzer.model_provenance = {}
    analyzer._ball_specs = {}
    analyzer.ball_batch_size = 8
    analyzer.ball_ensemble = False
    analyzer.ball_imgsz = override
    analyzer._init_ball_detector(str(weights))

    assert analyzer._ball_specs["primary"]["settings"]["imgsz"] == expected
    assert analyzer.model_provenance["ball_detector"]["imgsz"] == expected
