from pathlib import Path
import tempfile
import unittest

import numpy as np

from face_finder.scanner import best_similarity, image_files


class ScannerUtilitiesTests(unittest.TestCase):
    def test_best_similarity_checks_every_pair(self) -> None:
        references = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        candidates = [np.array([0.2, 0.98])]
        self.assertAlmostEqual(best_similarity(references, candidates), 0.98)

    def test_image_files_is_recursive_and_filters_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "nested"
            nested.mkdir()
            (root / "one.JPG").touch()
            (nested / "two.png").touch()
            (nested / "notes.txt").touch()
            self.assertCountEqual([item.name for item in image_files(root)], ["one.JPG", "two.png"])

    def test_image_files_excludes_output_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "matches"
            output.mkdir()
            (root / "keep.jpg").touch()
            (output / "skip.jpg").touch()
            self.assertEqual([item.name for item in image_files(root, [output])], ["keep.jpg"])


if __name__ == "__main__":
    unittest.main()
