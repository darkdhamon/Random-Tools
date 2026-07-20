import unittest
from pathlib import Path

from face_finder.app import record_unknown_photo


class BackgroundScanTests(unittest.TestCase):
    def test_unknown_person_is_ready_after_three_distinct_new_photos(self) -> None:
        groups: dict[int, set[Path]] = {}
        self.assertFalse(record_unknown_photo(groups, 7, Path("one.jpg"), 3))
        self.assertFalse(record_unknown_photo(groups, 7, Path("one.jpg"), 3))
        self.assertFalse(record_unknown_photo(groups, 7, Path("two.jpg"), 3))
        self.assertTrue(record_unknown_photo(groups, 7, Path("three.jpg"), 3))


if __name__ == "__main__":
    unittest.main()
