"""Local file paths and browser-style file URIs resolve identically."""

from pathlib import Path

from src.utils.path_utils import local_path_from_uri


def test_windows_file_uri_becomes_local_path():
    uri = (
        "file:///C:/Users/illya/Documents/volleyball_analitics/"
        "volleystat/data/video_for_test_29_07/test_4k.mp4"
    )

    assert local_path_from_uri(uri) == Path(
        "C:/Users/illya/Documents/volleyball_analitics/"
        "volleystat/data/video_for_test_29_07/test_4k.mp4"
    )


def test_file_uri_decodes_spaces():
    assert local_path_from_uri("file:///C:/match%20videos/set%201.mp4") == Path(
        "C:/match videos/set 1.mp4"
    )


def test_regular_path_is_unchanged():
    path = Path("data/video_for_test_29_07/test_4k.mp4")
    assert local_path_from_uri(path) == path
