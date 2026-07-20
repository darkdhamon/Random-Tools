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
        detector._vit_score = lambda _path: 0.90  # type: ignore[method-assign]
        result = detector.classify(Path("photo.jpg"))
        self.assertEqual(result.score, 0.90)
        self.assertEqual(
            [item["label"] for item in result.detections],
            ["WHOLE_IMAGE_NSFW", "FACE_FEMALE", "FEMALE_BREAST_EXPOSED"],
        )
        self.assertTrue(result.detections[0]["explicit"])
        self.assertFalse(result.detections[1]["explicit"])
        self.assertTrue(result.detections[2]["explicit"])

    def test_male_face_context_suppresses_low_confidence_female_breast_false_positive(self) -> None:
        detector = NsfwDetector.__new__(NsfwDetector)
        detector.detector = type(
            "MaleTorsoDetector",
            (),
            {"detect": lambda _self, _path: [
                {"class": "FACE_MALE", "score": 0.78},
                {"class": "FEMALE_BREAST_EXPOSED", "score": 0.57},
                {"class": "FEMALE_BREAST_EXPOSED", "score": 0.55},
            ]},
        )()
        detector._vit_score = lambda _path: 0.08  # type: ignore[method-assign]
        result = detector.classify(Path("shirtless-man.jpg"))
        self.assertEqual(result.score, 0.0)
        self.assertEqual(len(result.detections), 3)
        self.assertFalse(result.review_required)
        breast = next(item for item in result.detections if item["label"] == "FEMALE_BREAST_EXPOSED")
        self.assertFalse(breast["explicit"])

    def test_model_disagreement_requires_review_instead_of_hiding(self) -> None:
        detector = NsfwDetector.__new__(NsfwDetector)
        detector.detector = type("EmptyDetector", (), {"detect": lambda _self, _path: []})()
        detector._vit_score = lambda _path: 0.93  # type: ignore[method-assign]
        result = detector.classify(Path("uncertain.jpg"))
        self.assertEqual(result.score, 0.0)
        self.assertTrue(result.review_required)
        self.assertFalse(result.detections[0]["explicit"])


if __name__ == "__main__":
    unittest.main()
