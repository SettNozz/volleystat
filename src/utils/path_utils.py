"""Path normalization helpers for CLI and pipeline inputs."""

from pathlib import Path
from typing import Union
from urllib.parse import urlparse
from urllib.request import url2pathname


def local_path_from_uri(value: Union[str, Path]) -> Path:
    """Return a local path, decoding a ``file://`` URI when necessary.

    Browser and desktop file links commonly produce values such as
    ``file:///C:/videos/game.mp4``. Passing that string directly to
    :class:`pathlib.Path` creates ``file:\\C:\\...`` on Windows, which OpenCV
    cannot open.
    """
    raw = str(value)
    parsed = urlparse(raw)
    if parsed.scheme.lower() != "file":
        return Path(raw)

    if parsed.netloc and parsed.netloc.lower() != "localhost":
        uri_path = f"//{parsed.netloc}{parsed.path}"
    else:
        uri_path = parsed.path
    return Path(url2pathname(uri_path))
