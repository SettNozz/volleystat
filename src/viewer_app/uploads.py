"""Store videos dropped onto the viewer's New analysis page.

A browser never reveals the local path of a dropped file, so the viewer copies
its bytes into one upload directory and hands that path to the usual analysis
workflow. A second drop of the same file (same name and size) reuses the copy.
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path
from typing import BinaryIO, Optional

VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
CHUNK_BYTES = 8 << 20
# Leave room on the disk for the analysis outputs of the uploaded video.
FREE_SPACE_MARGIN_BYTES = 2 << 30


class UploadError(ValueError):
    """The dropped file cannot be stored; the message is shown to the user."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def safe_video_name(name: str) -> str:
    """Return a file name without directories or shell-unfriendly characters."""
    base = Path(str(name or "").replace("\\", "/")).name
    stem, suffix = os.path.splitext(base)
    if suffix.lower() not in VIDEO_SUFFIXES:
        allowed = ", ".join(sorted(VIDEO_SUFFIXES))
        raise UploadError(f"Unsupported video type '{suffix or base}'. Use one of: {allowed}")
    stem = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("._") or "video"
    return stem[:120] + suffix.lower()


def _candidates(upload_dir: Path, name: str):
    stem, suffix = os.path.splitext(name)
    yield upload_dir / name
    for index in range(1, 1000):
        yield upload_dir / f"{stem}_{index}{suffix}"


def find_existing_upload(upload_dir: Path, name: str, size: int) -> Optional[Path]:
    """Return an earlier upload with this name and byte size, if any."""
    safe = safe_video_name(name)
    for path in _candidates(upload_dir, safe):
        if not path.exists():
            return None
        if path.is_file() and path.stat().st_size == size:
            return path
    return None


def store_upload(upload_dir: Path, name: str, size: int, stream: BinaryIO) -> tuple[Path, bool]:
    """Copy ``size`` bytes from ``stream`` into ``upload_dir``.

    Returns ``(path, reused)``. The file appears under its final name only when
    every byte arrived, so an interrupted upload never looks like a video.
    """
    if size <= 0:
        raise UploadError("The dropped file is empty")
    existing = find_existing_upload(upload_dir, name, size)
    if existing is not None:
        return existing, True
    upload_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(upload_dir).free
    if free < size + FREE_SPACE_MARGIN_BYTES:
        raise UploadError(
            f"Not enough disk space: {size / 2**30:.1f} GB video, {free / 2**30:.1f} GB free",
            status=507,
        )
    target = next(p for p in _candidates(upload_dir, safe_video_name(name)) if not p.exists())
    partial = upload_dir / f".{target.name}.{uuid.uuid4().hex}.part"
    received = 0
    try:
        with open(partial, "wb") as handle:
            while received < size:
                chunk = stream.read(min(CHUNK_BYTES, size - received))
                if not chunk:
                    break
                handle.write(chunk)
                received += len(chunk)
        if received != size:
            raise UploadError(f"Upload interrupted after {received} of {size} bytes")
        os.replace(partial, target)
    finally:
        if partial.exists():
            partial.unlink()
    return target, False
