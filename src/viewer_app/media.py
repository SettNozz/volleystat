"""Video streaming with range requests and on-demand thumbnail/portrait decoding.

Moved verbatim from viewer.py, which re-exports these names.
"""

import math
import mimetypes
import os
from functools import lru_cache
from pathlib import Path

from flask import request, send_file, Response


def _stream_video(video_path: str):
    """Stream a video file with HTTP Range support for HTML5 player."""
    p = Path(video_path)
    if not p.exists():
        return ("Video not found", 404)

    mimetype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    file_size = os.path.getsize(video_path)
    range_header = request.headers.get("Range")

    if range_header:
        byte_start, byte_end = 0, file_size - 1
        match = range_header.replace("bytes=", "").split("-")
        byte_start = int(match[0])
        if match[1]:
            byte_end = int(match[1])
        chunk_size = byte_end - byte_start + 1

        def generate():
            with open(video_path, "rb") as f:
                f.seek(byte_start)
                remaining = chunk_size
                while remaining > 0:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {byte_start}-{byte_end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": chunk_size,
            "Content-Type": mimetype,
        }
        return Response(generate(), 206, headers=headers)

    return send_file(video_path, mimetype=mimetype)


# Rally clip thumbnails are decoded from the video on demand and kept only in
# memory: a result directory is never written to just to show a preview.
_THUMB_WIDTH = 320


def _decode_thumbnail(video_path: Path, frame: int, box=None):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = cap.read()
    finally:
        cap.release()
    if not ok or image is None:
        return None
    if box is not None:
        height, width = image.shape[:2]
        left, top, right, bottom = box
        # Keep the head and upper torso within the identity reference crop.
        bottom = min(bottom, top + 1.45 * (right - left))
        x1, y1 = max(0, math.floor(left)), max(0, math.floor(top))
        x2, y2 = min(width, math.ceil(right)), min(height, math.ceil(bottom))
        if x2 <= x1 or y2 <= y1:
            return None
        image = image[y1:y2, x1:x2]
    height, width = image.shape[:2]
    size = (_THUMB_WIDTH, max(1, round(height * _THUMB_WIDTH / width)))
    image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return encoded.tobytes() if ok else None


@lru_cache(maxsize=96)
def _decode_player_portrait(source, mtime, size, reference):
    return _decode_thumbnail(source, *reference)
