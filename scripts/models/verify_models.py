"""Check that every model file is present, pinned and backed up; restore missing ones.

    python scripts/models/verify_models.py              # report, exit 1 on a problem
    python scripts/models/verify_models.py --restore    # copy missing files back

For each entry of ``configs/model_registry.json`` this checks the local file
and its companion files against their pinned sha256, and looks in the MLflow
registry for a version holding exactly those bytes (``register_models.py
--all`` creates it). ``--restore`` brings back files that are *missing* from
that version. It never overwrites an existing file, and a file whose bytes
differ from its pin is only reported; deciding which bytes are right stays
with a person.

The registry is read through the running server when it answers, otherwise
straight from ``mlflow.db`` and ``mlartifacts/`` in the repository root, so a
restore also works with the server stopped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import model_registry as mr  # noqa: E402

LOCAL_DB = ROOT / "mlflow.db"
LOCAL_ARTIFACTS = ROOT / "mlartifacts"


def registry_client(db: Path = LOCAL_DB):
    """``(client, uri, served)``: the server when it answers, else the local store."""
    from mlflow.tracking import MlflowClient
    uri = mr.tracking_uri()
    try:
        mr._require_reachable(uri)
        client = MlflowClient(tracking_uri=uri, registry_uri=uri)
        client.search_registered_models(max_results=1)
        return client, uri, uri.startswith(("http://", "https://"))
    except Exception:
        if not db.is_file():
            return None, None, False
        local = "sqlite:///" + str(db).replace("\\", "/")
        return MlflowClient(tracking_uri=local, registry_uri=local), local, False


def backup_version(client, entry: Dict[str, Any]):
    """The registry version whose sha256 tag equals the entry's pin, if any."""
    pinned = entry.get("sha256")
    if client is None or not pinned:
        return None
    try:
        versions = client.search_model_versions(f"name='{entry.get('registry_name')}'")
    except Exception:
        return None
    matches = [v for v in versions if (v.tags or {}).get("sha256") == pinned]
    return max(matches, key=lambda v: int(v.version)) if matches else None


def artifact_uri(version, entry: Dict[str, Any], relative: Optional[str]) -> Optional[str]:
    """Where the registry keeps the main file (``relative=None``) or a companion."""
    base = version.source.rstrip("/")
    if relative is None:
        return f"{base}/{entry.get('artifact') or mr.fallback_path(entry).name}"
    companions = json.loads((version.tags or {}).get("companions") or "{}")
    if relative not in companions:
        return None
    return f"{base}/{companions[relative][0]}"


def fetch(uri: str, destination: Path, tracking: Optional[str], served: bool,
          artifacts_root: Path = LOCAL_ARTIFACTS) -> Path:
    """Download one artifact into ``destination`` (a directory) and return the file."""
    prefix = "mlflow-artifacts:/"
    if uri.startswith(prefix) and not served:
        source = artifacts_root / uri[len(prefix):].lstrip("/")
        if not source.is_file():
            raise FileNotFoundError(f"backup file not found: {source}")
        target = destination / source.name
        shutil.copy2(source, target)
        return target
    from mlflow.artifacts import download_artifacts
    return Path(download_artifacts(artifact_uri=uri, dst_path=str(destination), tracking_uri=tracking))


def verify(config: Dict[str, Dict[str, Any]], restore: bool = False, client_info=None,
           artifacts_root: Path = LOCAL_ARTIFACTS) -> int:
    client, tracking, served = client_info if client_info is not None else registry_client()
    print(f"Registry: {tracking or 'not available (no server, no mlflow.db)'}\n")
    problems = 0
    for name, entry in config.items():
        required = entry.get("required", True)
        files = mr.check_files(entry)
        version = backup_version(client, entry)
        backup = f"backup v{version.version}" if version else "NOT BACKED UP"
        if not entry.get("sha256"):
            backup = "no pin"
        elif entry.get("backup") is False:
            backup = f"no registry copy by design; source {entry.get('source_url') or 'unknown'}"
        print(f"{name}  [{backup}]{'' if required else '  (optional)'}")
        for row in files:
            status = row["status"]
            if status == "missing" and restore and version is not None and row["pinned"]:
                status = restore_file(version, entry, row, tracking, served, artifacts_root)
            mark = {"ok": "ok", "restored": "RESTORED"}.get(status, status.upper())
            print(f"    {mark:10s} {row['name']}  {row['path']}")
            if status not in ("ok", "restored", "unpinned") and required:
                problems += 1
        if required and entry.get("sha256") and version is None and entry.get("backup") is not False:
            problems += 1  # no registry, or no version with these bytes: not recoverable
    if problems:
        print(f"\n{problems} problem(s). Missing files: --restore (needs a backup); "
              "not backed up: python scripts/models/register_models.py --all")
    else:
        print("\nAll required model files match their pins and are backed up.")
    return 1 if problems else 0


def restore_file(version, entry, row, tracking, served, artifacts_root) -> str:
    uri = artifact_uri(version, entry, None if row["role"] == "main" else row["name"])
    if uri is None:
        return "missing (not in backup)"
    with tempfile.TemporaryDirectory(dir=str(ROOT)) as scratch:
        try:
            fetched = fetch(uri, Path(scratch), tracking, served, artifacts_root)
        except Exception as exc:
            return f"missing (restore failed: {str(exc).splitlines()[0][:80]})"
        if mr.sha256_of(fetched) != row["pinned"]:
            return "missing (backup has other bytes)"
        target: Path = row["path"]
        if target.exists():  # appeared meanwhile: never overwrite
            return "missing (target appeared, left alone)"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fetched, target)
    return "restored"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--restore", action="store_true", help="Copy missing pinned files back from the registry")
    ap.add_argument("--name", action="append", default=[], help="Only these logical names (repeatable)")
    args = ap.parse_args()
    config = mr.load_config()
    if args.name:
        unknown = set(args.name) - set(config)
        if unknown:
            ap.error(f"unknown model(s): {', '.join(sorted(unknown))}")
        config = {k: v for k, v in config.items() if k in args.name}
    return verify(config, restore=args.restore)


if __name__ == "__main__":
    raise SystemExit(main())
