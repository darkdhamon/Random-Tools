import unittest
from pathlib import Path

from face_finder.nsfw import NsfwDetector


class FakeDetector:
    def detect(self, _path: str) -> list[dict[str, object]]:
        return [
            {"class": "FACE_FEMALE", "score": 0.91},
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.76},
            {"class": "FEET_COVERED", "score": 0.10},
        ]


class NsfwTests(unittest.TestCase):
    def test_classification_preserves_findings_and_marks_score_contributors(self) -> None:
        detector = NsfwDetector.__new__(NsfwDetector)
        detector.detector = FakeDetector()
        result = detector.classify(Path("photo.jpg"))
        self.assertEqual(result.score, 0.76)
        self.assertEqual([item["label"] for item in result.detections], ["FACE_FEMALE", "FEMALE_BREAST_EXPOSED"])
        self.assertFalse(result.detections[0]["explicit"])
        self.assertTrue(result.detections[1]["explicit"])


if __name__ == "__main__":
    unittest.main()
