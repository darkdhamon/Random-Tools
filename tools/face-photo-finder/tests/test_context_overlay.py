import unittest

import numpy as np

from face_finder.app import detection_context
from face_finder.scanner import DetectedFace


class ContextOverlayTests(unittest.TestCase):
    def test_every_detected_face_gets_the_expected_review_status(self) -> None:
        preview = np.zeros((112, 112, 3), dtype=np.uint8)
        embedding = np.zeros(128, dtype=np.float32)
        faces = [
            DetectedFace(embedding, preview, (10, 10, 20, 20)),
            DetectedFace(embedding, preview, (40, 10, 20, 20)),
            DetectedFace(embedding, preview, (70, 10, 20, 20)),
            DetectedFace(embedding, preview, (100, 10, 20, 20)),
        ]
        states = {
            0: ("identified", "Alex"),
            1: ("unknown", "Unknown person"),
            3: ("not_face", "Not a face"),
        }
        annotations = detection_context(faces, current_index=2, states=states)
        self.assertEqual(
            [(item.status, item.label) for item in annotations],
            [("identified", "Alex"), ("unknown", "Unknown person"), ("current", "Person to identify")],
        )

    def test_future_faces_are_red_unprocessed_targets(self) -> None:
        preview = np.zeros((112, 112, 3), dtype=np.uint8)
        embedding = np.zeros(128, dtype=np.float32)
        faces = [
            DetectedFace(embedding, preview, (10, 10, 20, 20)),
            DetectedFace(embedding, preview, (40, 10, 20, 20)),
        ]
        annotations = detection_context(faces, current_index=0, states={})
        self.assertEqual(annotations[0].status, "current")
        self.assertEqual(annotations[1].status, "unprocessed")


if __name__ == "__main__":
    unittest.main()
