"""Put the local weight files into the local MLflow Model Registry.

Run once per machine, and again whenever a training run produces something the
pipeline should start using:

    python scripts/models/run_mlflow_server.py          # in another terminal
    python scripts/models/register_models.py --all
    python scripts/models/register_models.py --name event_sequence \
        --path action_recognition/models/<new run>/final_event_sequence.pt \
        --note "gold 3x, seed 42, macro_f1 0.53"

Each weight file is copied into the registry as a run artifact, so the version
is immutable even if the file on disk is later overwritten or deleted. The
sha256 is stored as a tag, which makes the command idempotent: registering the
same bytes twice reuses the existing version instead of adding a new one.

``--alias`` (default ``production``) is what the pipeline resolves, so promoting
a model is a one-line operation and demoting is `--alias staging`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.model_registry import (  # noqa: E402
    CONFIG_PATH, load_config, sha256_of, tracking_uri,
)

EXPERIMENT = "model_registry"


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        return None


def _existing_version_with_sha(client, registry_name: str, sha: str):
    try:
        versions = client.search_model_versions(f"name='{registry_name}'")
    except Exception:
        return None
    for version in versions:
        if (version.tags or {}).get("sha256") == sha:
            return version
    return None


def register_one(
    name: str,
    entry: Dict[str, Any],
    path: Path,
    alias: str,
    note: Optional[str],
    force: bool,
) -> Dict[str, Any]:
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)

    registry_name = entry.get("registry_name", name)
    sha = sha256_of(path)
    size_mb = round(path.stat().st_size / 1e6, 1)

    existing = None if force else _existing_version_with_sha(client, registry_name, sha)
    if existing is not None:
        client.set_registered_model_alias(registry_name, alias, existing.version)
        return {"name": name, "registry_name": registry_name, "version": existing.version,
                "status": "already registered (same sha256), alias moved",
                "size_mb": size_mb, "sha256": sha[:12]}

    mlflow.set_experiment(EXPERIMENT)
    tags = {
        "model": name,
        "sha256": sha,
        "source_path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "size_mb": str(size_mb),
    }
    if _git_commit():
        tags["git_commit"] = _git_commit()
    for key in ("imgsz", "tile_grid", "model_type"):
        if entry.get(key) is not None:
            tags[key] = json.dumps(entry[key])
    if note or entry.get("note"):
        tags["note"] = note or entry["note"]

    with mlflow.start_run(run_name=f"register:{name}", tags=tags) as run:
        # artifact_path=name keeps the file addressable as <version.source>/<file>
        mlflow.log_artifact(str(path), artifact_path=name)
        run_id = run.info.run_id
        artifact_uri = mlflow.get_artifact_uri(name)

    try:
        client.create_registered_model(registry_name)
    except Exception:
        pass  # already exists

    # create_model_version, not mlflow.register_model: these are raw weight
    # files, not MLflow "logged models", and since 3.x the fluent helper
    # refuses anything that is not one.
    version = client.create_model_version(
        name=registry_name, source=artifact_uri, run_id=run_id, tags=tags,
    )
    client.set_registered_model_alias(registry_name, alias, version.version)
    for key, value in tags.items():
        try:
            client.set_model_version_tag(registry_name, version.version, key, value)
        except Exception:
            pass
    return {"name": name, "registry_name": registry_name, "version": version.version,
            "status": f"registered, alias '{alias}'", "size_mb": size_mb, "sha256": sha[:12]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="Register every configured model")
    ap.add_argument("--name", action="append", default=[],
                    help="Logical name from configs/model_registry.json (repeatable)")
    ap.add_argument("--path", default=None,
                    help="Weight file to register instead of the configured fallback "
                         "(only valid with a single --name)")
    ap.add_argument("--alias", default="production",
                    help="Alias the pipeline resolves (default: production)")
    ap.add_argument("--note", default=None, help="Free text stored with the version")
    ap.add_argument("--force", action="store_true",
                    help="Register a new version even if these exact bytes are already in")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only report what would be registered")
    args = ap.parse_args()

    config = load_config()
    names = list(config) if args.all else args.name
    if not names:
        ap.error("pass --all or at least one --name "
                 f"(configured: {', '.join(sorted(config))})")
    if args.path and len(names) != 1:
        ap.error("--path applies to exactly one --name")

    print(f"Tracking URI: {tracking_uri()}")
    print(f"Config      : {CONFIG_PATH.relative_to(ROOT)}\n")

    rows, missing = [], []
    for name in names:
        entry = config.get(name)
        if entry is None:
            print(f"  ! unknown model '{name}' — not in the config")
            continue
        raw = args.path or entry.get("fallback")
        if not raw:
            print(f"  ! {name}: no path configured and none given")
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            missing.append((name, path))
            print(f"  - {name}: file not found, skipped ({path})")
            continue
        if args.dry_run:
            print(f"  = {name}: would register {path} "
                  f"({round(path.stat().st_size / 1e6, 1)} MB)")
            continue
        try:
            rows.append(register_one(name, entry, path, args.alias, args.note, args.force))
            row = rows[-1]
            print(f"  + {row['name']:20s} v{row['version']:<3} {row['size_mb']:>7.1f} MB  "
                  f"{row['sha256']}  {row['status']}")
        except Exception as exc:
            print(f"  ! {name}: {type(exc).__name__}: {str(exc).splitlines()[0][:150]}")

    if missing:
        print("\nNot registered because the file is absent — train or download it first:")
        for name, path in missing:
            print(f"  {name}: {path}")
    if rows:
        print(f"\n{len(rows)} model(s) in the registry. The pipeline now resolves them by "
              f"alias '{args.alias}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
