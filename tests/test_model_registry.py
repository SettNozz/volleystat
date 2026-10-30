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
    return register_one(name, entry, path, alias, kw.get("note"), kw.get("force", False))


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
