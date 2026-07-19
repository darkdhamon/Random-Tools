from pathlib import Path
import unittest

import numpy as np

from face_finder.app import ForcedIdentityMatch, IdentityRequest, forced_identity_decision


class BatchIdentityTests(unittest.TestCase):
    def test_possible_match_can_be_included_or_excluded(self) -> None:
        path = Path("possible.jpg")
        embedding = np.array([1.0, 0.0], dtype=np.float32)
        included = ForcedIdentityMatch(embedding, 4, False, False)
        self.assertIs(forced_identity_decision(path, embedding, {path: [included]}), included)
        excluded = ForcedIdentityMatch(embedding, 4, True, False)
        self.assertTrue(forced_identity_decision(path, embedding, {path: [excluded]}).excluded)  # type: ignore[union-attr]

    def test_unrelated_face_does_not_inherit_batch_identity(self) -> None:
        path = Path("possible.jpg")
        decision = ForcedIdentityMatch(np.array([1.0, 0.0]), 4, False, False)
        self.assertIsNone(
            forced_identity_decision(path, np.array([0.0, 1.0]), {path: [decision]})
        )

    def test_related_candidate_exclusion_toggles(self) -> None:
        request = IdentityRequest(Path("current.jpg"), np.zeros((8, 8, 3), dtype=np.uint8), [])
        request.add_related_face(
            Path("possible.jpg"), np.array([1.0, 0.0]), np.zeros((8, 8, 3), dtype=np.uint8)
        )
        candidate = request.related_faces[0]
        candidate.excluded = not candidate.excluded
        self.assertTrue(candidate.excluded)
        candidate.excluded = not candidate.excluded
        self.assertFalse(candidate.excluded)


if __name__ == "__main__":
    unittest.main()
