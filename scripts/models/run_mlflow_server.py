"""Start the local MLflow server that holds the model registry.

The registry needs a database behind it — the plain ``mlruns/`` file store
cannot serve model versions at all — so this runs MLflow against a sqlite file
in the repository root and serves artifacts itself, which is what lets the
pipeline download weights over the same connection.

    python scripts/models/run_mlflow_server.py          # http://127.0.0.1:5000

Existing file-based runs under ``mlruns/`` are *not* visible in this server:
they live in a different store. That is deliberate — old experiment history
stays where it is, while models are served from one place. To browse the old
runs use the previous command (``mlflow ui --workers 1``) in another terminal.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "mlflow.db"
DEFAULT_ARTIFACTS = ROOT / "mlartifacts"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    ap.add_argument("--workers", type=int, default=1,
                    help="Windows needs 1 (WinError 10022 with more)")
    args = ap.parse_args()

    args.artifacts.mkdir(parents=True, exist_ok=True)
    # sqlite URIs need forward slashes even on Windows.
    db_uri = "sqlite:///" + str(args.db).replace("\\", "/")

    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", db_uri,
        "--artifacts-destination", str(args.artifacts),
        "--serve-artifacts",
        "--host", args.host,
        "--port", str(args.port),
        "--workers", str(args.workers),
    ]
    print("Model registry:", db_uri)
    print("Artifacts     :", args.artifacts)
    print("UI            :", f"http://{args.host}:{args.port}")
    print("Pipeline reads:", f"MLFLOW_TRACKING_URI=http://{args.host}:{args.port} "
                             "(this is the default)\n")
    print(" ".join(cmd), "\n")
    try:
        return subprocess.call(cmd, cwd=str(ROOT))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
