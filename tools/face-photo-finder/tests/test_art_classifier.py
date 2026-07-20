import unittest

import numpy as np

from face_finder.art_classifier import classification_from_probabilities, face_context


class ArtClassifierTests(unittest.TestCase):
    def test_high_confidence_statue_is_excluded(self) -> None:
        result = classification_from_probabilities(np.array([0.05, 0.05, 0.90]))
        self.assertEqual(result.kind, "statue")
        self.assertTrue(result.exclude_from_biometrics)

    def test_uncertain_artwork_is_kept_for_review(self) -> None:
        result = classification_from_probabilities(np.array([0.30, 0.65, 0.05]))
        self.assertEqual(result.kind, "artwork")
        self.assertFalse(result.exclude_from_biometrics)

    def test_photographic_person_is_not_excluded(self) -> None:
        result = classification_from_probabilities(np.array([0.96, 0.02, 0.02]))
        self.assertEqual(result.kind, "photo")
        self.assertFalse(result.exclude_from_biometrics)

    def test_face_context_expands_and_clamps_box(self) -> None:
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        crop = face_context(image, (0, 10, 40, 40))
        self.assertEqual(crop.shape, (80, 70, 3))


if __name__ == "__main__":
    unittest.main()
