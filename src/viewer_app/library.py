"""Pure helpers for the results library index.

Moved verbatim from viewer.py, which re-exports these names.
"""

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Results library
# ---------------------------------------------------------------------------

_ORIGINAL_VIDEO_RE = re.compile(r'"original_video_path"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _annotation_original_video(path: Path) -> str:
    """Read the recorded source video without parsing a multi-megabyte file.

    Analysis writes ``original_video_path`` into the annotation header, well
    inside the first kilobytes, so the library never loads the frame array.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(8192).decode("utf-8", errors="ignore")
    except OSError:
        return ""
    match = _ORIGINAL_VIDEO_RE.search(head)
    if not match:
        return ""
    try:
        return json.loads(f'"{match.group(1)}"')
    except json.JSONDecodeError:
        return ""


def _library_summary(stats_path: Path) -> dict | None:
    """Summarise one result for the library card."""
    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(stats, dict):
        return None

    fps = float(stats.get("fps") or 0.0)
    total_frames = int(stats.get("total_frames") or 0)
    lifecycle = stats.get("rally_lifecycle")
    lifecycle_available = isinstance(lifecycle, dict)
    rallies = (
        int(lifecycle.get("playable_rallies", 0) or 0) if lifecycle_available
        else int(stats.get("total_rallies", 0) or 0)
    )
    return {
        "video": stats.get("video", ""),
        "fps": round(fps, 2) if fps else None,
        "total_frames": total_frames or None,
        "duration_sec": round(total_frames / fps, 1) if fps > 0 and total_frames else None,
        "ball_detection_rate": stats.get("ball_detection_rate"),
        "total_rallies": rallies,
        "total_events": len(stats.get("all_events") or []),
        "processing_time_sec": stats.get("processing_time_sec"),
        "has_lifecycle": lifecycle_available,
        "has_match_analytics": bool(stats.get("match_analytics")),
        "has_reconstruction": bool(stats.get("ball_reconstruction")),
    }
