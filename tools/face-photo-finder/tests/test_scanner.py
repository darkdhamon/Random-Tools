from pathlib import Path
import tempfile
import unittest

import numpy as np

from face_finder.scanner import DetectedFace, best_similarity, build_reference_embeddings, image_files


class ScannerUtilitiesTests(unittest.TestCase):
    def test_reference_picker_can_select_multiple_faces(self) -> None:
        faces = [
            DetectedFace(np.array([1.0, 0.0]), np.zeros((2, 2, 3), dtype=np.uint8)),
            DetectedFace(np.array([0.0, 1.0]), np.zeros((2, 2, 3), dtype=np.uint8)),
            DetectedFace(np.array([0.5, 0.5]), np.zeros((2, 2, 3), dtype=np.uint8)),
        ]

        class FakeEngine:
            def detect_faces(self, _path: Path) -> list[DetectedFace]:
                return faces

        selected = build_reference_embeddings(
            FakeEngine(),  # type: ignore[arg-type]
            [Path("group.jpg")],
            lambda _path, _faces: [0, 2],
        )
        self.assertEqual(len(selected), 2)
        np.testing.assert_array_equal(selected[0], faces[0].embedding)
        np.testing.assert_array_equal(selected[1], faces[2].embedding)

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
