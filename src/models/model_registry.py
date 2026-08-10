"""One place that answers "which weights is this run using?".

The repository accumulated weights faster than it accumulated decisions about
them: `models/` holds several YOLO training runs, `action_recognition/models/`
holds twenty-odd checkpoint directories, and the net detector defaults to a SAM
checkpoint in ~/Downloads while a different SAM checkpoint sits in `models/`.
Every consumer hard-coded its own path, so the only way to know what produced a
result was to read the source at the commit it ran on.

This module makes the local MLflow Model Registry that source of truth. A
caller asks for a *logical* name — ``ball_detector``, ``pose_estimator`` — and
gets back a concrete file plus the provenance of that file, which the pipeline
writes into ``*_stats.json``. Nothing else in the codebase needs to know where
weights live.

Three rules make it usable rather than merely correct:

* **The registry is authoritative, not mandatory.** If MLflow is not installed,
  the server is down, or a name was never registered, resolution falls back to
  the configured local path and says so loudly — in the log and in the
  provenance record. An unreachable tracking server must not stop a video from
  being analysed.
* **A downloaded version is cached by version number.** Artifacts are copied
  once into ``models/_registry_cache/<name>/<version>/``; later runs reuse them.
  If the server is unreachable but a cache exists, the cache wins over the
  fallback — a known version beats an unknown file.
* **An explicit path always wins.** ``--ball-model`` and friends are how you
  test something that is deliberately not in the registry yet; they bypass this
  module entirely and are recorded as ``override``.

Configuration lives in ``configs/model_registry.json`` so adding a model does
not require touching code.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

log = get_logger("ModelRegistry")

_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = _ROOT / "configs" / "model_registry.json"
CACHE_DIR = _ROOT / "models" / "_registry_cache"

# A local server, as configured by scripts/models/run_mlflow_server.py.
# MLFLOW_TRACKING_URI overrides it, which is what CI and the tests use to point
# at a temporary sqlite store instead.
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"

# A dead server must cost seconds, not minutes: analysis has to start anyway.
_HTTP_TIMEOUT_SEC = "5"
_HTTP_RETRIES = "1"


@dataclass
class ResolvedModel:
    """A concrete weight file plus where the decision to use it came from."""

    name: str
    path: Path
    source: str                      # registry | cache | fallback | override
    version: Optional[str] = None
    alias: Optional[str] = None
    run_id: Optional[str] = None
    sha256: Optional[str] = None
    warning: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def from_registry(self) -> bool:
        return self.source in {"registry", "cache"}

    def as_dict(self) -> Dict[str, Any]:
        """Provenance small enough to store in every result file."""
        out: Dict[str, Any] = {
            "name": self.name,
            "source": self.source,
            "path": str(self.path),
        }
        if self.version:
            out["version"] = self.version
        if self.alias:
            out["alias"] = self.alias
        if self.run_id:
            out["run_id"] = self.run_id
        if self.sha256:
            out["sha256"] = self.sha256[:12]
        if self.warning:
            out["warning"] = self.warning
        out.update(self.extra)
        return out

    def __str__(self) -> str:
        where = f"{self.source}"
        if self.version:
            where += f" v{self.version}"
        return f"{self.name} [{where}] {self.path}"


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


def load_config(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    path = Path(path or CONFIG_PATH)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("models", data)


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------------
# Resolution
# ----------------------------------------------------------------------


def _cached_versions(name: str) -> List[str]:
    directory = CACHE_DIR / name
    if not directory.is_dir():
        return []
    return sorted(
        (p.name for p in directory.iterdir() if p.is_dir()),
        key=lambda v: int(v) if v.isdigit() else -1,
    )


def _cached_file(name: str, version: str, artifact: Optional[str]) -> Optional[Path]:
    directory = CACHE_DIR / name / str(version)
    if not directory.is_dir():
        return None
    if artifact:
        exact = directory / Path(artifact).name
        if exact.exists():
            return exact
    files = [p for p in directory.rglob("*") if p.is_file()]
    weights = [p for p in files if p.suffix in {".pt", ".pth", ".onnx", ".task", ".tflite"}]
    candidates = weights or files
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def _fallback(
    name: str, entry: Dict[str, Any], reason: str, allow_fallback: bool
) -> ResolvedModel:
    fallback = entry.get("fallback")
    if not fallback:
        raise FileNotFoundError(
            f"model '{name}' is not in the registry ({reason}) and has no fallback path"
        )
    path = Path(fallback).expanduser()
    if not path.is_absolute():
        path = _ROOT / path
    if not allow_fallback:
        raise FileNotFoundError(f"model '{name}' unavailable from the registry: {reason}")
    warning = f"registry unavailable ({reason}); using local file {path}"
    log.warning(f"{name}: {warning}")
    return ResolvedModel(
        name=name, path=path, source="fallback", warning=warning,
        extra={k: v for k, v in entry.items() if k in ("imgsz", "tile_grid", "model_type")},
    )


def resolve_model(
    name: str,
    override: Optional[str] = None,
    config: Optional[Dict[str, Dict[str, Any]]] = None,
    allow_fallback: bool = True,
    use_registry: bool = True,
) -> ResolvedModel:
    """Return the weight file to load for a logical model name.

    Never raises for the ordinary failure modes (no MLflow, no server, name not
    registered): those degrade to the configured local file, with the reason
    recorded on the result.
    """
    entries = config if config is not None else load_config()
    entry = dict(entries.get(name) or {})
    extra = {k: v for k, v in entry.items() if k in ("imgsz", "tile_grid", "model_type")}

    if override:
        path = Path(override)
        if not path.is_absolute():
            path = _ROOT / path
        log.info(f"{name}: explicit path override {path}")
        return ResolvedModel(name=name, path=path, source="override", extra=extra)

    if not entry:
        raise KeyError(
            f"unknown model '{name}'. Add it to {CONFIG_PATH.relative_to(_ROOT)} "
            f"(known: {', '.join(sorted(entries)) or 'none'})"
        )

    if not use_registry or os.environ.get("VOLLEYSTAT_NO_REGISTRY"):
        return _fallback(name, entry, "registry disabled", allow_fallback)

    registry_name = entry.get("registry_name", name)
    alias = entry.get("alias", "production")
    artifact = entry.get("artifact")

    try:
        import mlflow  # noqa: F401
        from mlflow.tracking import MlflowClient
    except Exception as exc:  # ImportError, or a broken install
        return _fallback(name, entry, f"mlflow not importable: {exc}", allow_fallback)

    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", _HTTP_TIMEOUT_SEC)
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", _HTTP_RETRIES)
    uri = tracking_uri()

    try:
        client = MlflowClient(tracking_uri=uri, registry_uri=uri)
        version = _lookup_version(client, registry_name, alias)
    except Exception as exc:
        cached = _cached_versions(name)
        if cached:
            # A known version already on disk is a better answer than an
            # unrelated file: the server being down does not change which
            # weights this project decided on.
            newest = cached[-1]
            path = _cached_file(name, newest, artifact)
            if path:
                warning = f"registry unreachable ({_short(exc)}); using cached version {newest}"
                log.warning(f"{name}: {warning}")
                return ResolvedModel(
                    name=name, path=path, source="cache", version=newest,
                    alias=alias, warning=warning, extra=extra,
                )
        return _fallback(name, entry, _short(exc), allow_fallback)

    if version is None:
        return _fallback(
            name, entry, f"'{registry_name}' has no version with alias '{alias}'",
            allow_fallback,
        )

    cached_path = _cached_file(name, version.version, artifact)
    if cached_path is not None:
        log.info(f"{name}: registry v{version.version} (cached) {cached_path.name}")
        return ResolvedModel(
            name=name, path=cached_path, source="registry", version=str(version.version),
            alias=alias, run_id=version.run_id, extra=extra,
        )

    try:
        path = _download(client, version, name, artifact)
    except Exception as exc:
        return _fallback(name, entry, f"download failed: {_short(exc)}", allow_fallback)

    log.info(f"{name}: registry v{version.version} -> {path}")
    return ResolvedModel(
        name=name, path=path, source="registry", version=str(version.version),
        alias=alias, run_id=version.run_id, extra=extra,
    )


def _short(exc: Exception) -> str:
    text = str(exc).strip().splitlines()
    return (text[0] if text else exc.__class__.__name__)[:160]


def _lookup_version(client, registry_name: str, alias: str):
    """Prefer the alias; fall back to the newest version of that model."""
    try:
        return client.get_model_version_by_alias(registry_name, alias)
    except Exception:
        pass
    versions = client.search_model_versions(f"name='{registry_name}'")
    if not versions:
        return None
    return max(versions, key=lambda v: int(v.version))


def _download(client, version, name: str, artifact: Optional[str]) -> Path:
    from mlflow.artifacts import download_artifacts

    target = CACHE_DIR / name / str(version.version)
    target.mkdir(parents=True, exist_ok=True)
    uri = version.source if not artifact else f"{version.source.rstrip('/')}/{artifact}"
    local = Path(
        download_artifacts(artifact_uri=uri, dst_path=str(target),
                           tracking_uri=tracking_uri())
    )
    if local.is_dir():
        found = _cached_file(name, str(version.version), artifact)
        if found is None:
            raise FileNotFoundError(f"no weight file inside {local}")
        return found
    return local


def clear_cache(name: Optional[str] = None) -> int:
    """Drop cached artifacts. Returns how many version directories went."""
    root = CACHE_DIR / name if name else CACHE_DIR
    if not root.exists():
        return 0
    removed = 0
    for version_dir in sorted(root.rglob("*")):
        if version_dir.is_dir() and version_dir.parent.parent == CACHE_DIR:
            shutil.rmtree(version_dir, ignore_errors=True)
            removed += 1
    return removed
