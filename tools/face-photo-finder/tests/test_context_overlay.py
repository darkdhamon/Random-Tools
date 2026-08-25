import unittest
from pathlib import Path

import numpy as np

from face_finder.app import ContextFace, IdentityRequest, detection_context
from face_finder.scanner import DetectedFace


class ContextOverlayTests(unittest.TestCase):
    def test_identity_request_starts_with_current_target_and_allows_multiple(self) -> None:
        preview = np.zeros((32, 32, 3), dtype=np.uint8)
        context = [
            ContextFace((1, 1, 10, 10), "current", "Current", "detected:0"),
            ContextFace((20, 1, 10, 10), "unprocessed", "Unprocessed", "detected:1"),
        ]
        request = IdentityRequest(Path("group.jpg"), preview, [], context_faces=context)
        self.assertEqual(request.selected_face_keys, {"detected:0"})
        request.selected_face_keys.add("detected:1")
        self.assertEqual(request.selected_face_keys, {"detected:0", "detected:1"})

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
