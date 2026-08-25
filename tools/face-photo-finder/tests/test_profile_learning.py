import unittest

import numpy as np

from face_finder.app import add_known_sample, add_unknown_sample
from face_finder.catalog import KnownIdentity, UnknownGroup, best_known_identity, best_unknown_group


class LiveProfileLearningTests(unittest.TestCase):
    def test_automatic_named_sample_is_available_immediately(self) -> None:
        original = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        new_angle = np.array([0.7, 0.714, 0.0], dtype=np.float32)
        later_angle = np.array([0.68, 0.73, 0.0], dtype=np.float32)
        identities = [KnownIdentity(1, "Person", (original,))]
        identities = add_known_sample(identities, 1, "Person", new_angle)
        identity_id, score = best_known_identity(later_angle, identities, 0.55)
        self.assertEqual(identity_id, 1)
        self.assertGreater(score, 0.98)

    def test_automatic_anonymous_sample_is_available_immediately(self) -> None:
        original = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        new_angle = np.array([0.7, 0.714, 0.0], dtype=np.float32)
        later_angle = np.array([0.68, 0.73, 0.0], dtype=np.float32)
        groups = [UnknownGroup(7, (original,))]
        groups = add_unknown_sample(groups, 7, new_angle)
        group_id, score = best_unknown_group(later_angle, groups, 0.55)
        self.assertEqual(group_id, 7)
        self.assertGreater(score, 0.98)


if __name__ == "__main__":
    unittest.main()
