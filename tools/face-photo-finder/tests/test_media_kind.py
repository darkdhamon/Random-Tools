import unittest
from pathlib import Path

from face_finder.media_kind import filename_media_kind


class MediaKindTests(unittest.TestCase):
    def test_filename_classifies_screenshots_and_documents(self) -> None:
        self.assertEqual(filename_media_kind(Path("Screenshot 2026-01-01.png")), "screenshot")
        self.assertEqual(filename_media_kind(Path("receipt_store.jpg")), "document")
        self.assertIsNone(filename_media_kind(Path("family-vacation.jpg")))


if __name__ == "__main__":
    unittest.main()
