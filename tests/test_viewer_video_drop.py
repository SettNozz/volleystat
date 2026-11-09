import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import viewer
from src.viewer_app.uploads import UploadError, find_existing_upload, safe_video_name, store_upload


class SafeNameTests(unittest.TestCase):
    def test_strips_directories_and_odd_characters(self):
        self.assertEqual(safe_video_name("C:\\clips\\..\\my game (1).MP4"), "my_game_1.mp4")
        self.assertEqual(safe_video_name("../../etc/x.webm"), "x.webm")

    def test_rejects_non_video(self):
        with self.assertRaises(UploadError):
            safe_video_name("notes.txt")


class StoreUploadTests(unittest.TestCase):
    def test_reuses_same_name_and_size_and_suffixes_a_different_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "uploads"
            first, reused = store_upload(root, "game.mp4", 4, io.BytesIO(b"abcd"))
            self.assertFalse(reused)
            self.assertEqual(first.read_bytes(), b"abcd")
            again, reused = store_upload(root, "game.mp4", 4, io.BytesIO(b"zzzz"))
            self.assertTrue(reused)
            self.assertEqual(again, first)
            other, reused = store_upload(root, "game.mp4", 3, io.BytesIO(b"xyz"))
            self.assertFalse(reused)
            self.assertEqual(other.name, "game_1.mp4")
            self.assertEqual(find_existing_upload(root, "game.mp4", 3), other)
            self.assertIsNone(find_existing_upload(root, "game.mp4", 99))

    def test_interrupted_upload_leaves_no_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(UploadError):
                store_upload(root, "cut.webm", 10, io.BytesIO(b"short"))
            self.assertEqual(list(root.iterdir()), [])


class UploadRouteTests(unittest.TestCase):
    def test_drop_then_check_returns_the_stored_path(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(viewer, "UPLOAD_DIR", Path(tmp)):
            with viewer.app.test_client() as client:
                self.assertIsNone(client.get("/api/upload_video?name=a.mp4&size=3").get_json()["path"])
                response = client.post(
                    "/api/upload_video", data=b"vid",
                    headers={"X-File-Name": "a%20b.mp4", "Content-Type": "application/octet-stream"},
                )
                self.assertEqual(response.status_code, 200, response.get_json())
                stored = Path(response.get_json()["path"])
                self.assertEqual(stored, Path(tmp) / "a_b.mp4")
                self.assertEqual(stored.read_bytes(), b"vid")
                check = client.get("/api/upload_video?name=a b.mp4&size=3").get_json()
                self.assertEqual(Path(check["path"]), stored)

    def test_rejects_non_video_upload(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(viewer, "UPLOAD_DIR", Path(tmp)):
            with viewer.app.test_client() as client:
                response = client.post("/api/upload_video", data=b"x", headers={"X-File-Name": "a.txt"})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
