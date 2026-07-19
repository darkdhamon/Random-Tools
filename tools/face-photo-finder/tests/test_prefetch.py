from pathlib import Path
import threading
import time
import unittest

import numpy as np

from face_finder.prefetch import DetectionPrefetcher
from face_finder.scanner import DetectedFace


class FakeEngine:
    def detect_faces(self, path: Path) -> list[DetectedFace]:
        value = float(int(path.stem))
        return [
            DetectedFace(
                np.array([value, 0.0], dtype=np.float32),
                np.zeros((16, 16, 3), dtype=np.uint8),
                (1, 1, 10, 10),
            )
        ]


class DetectionPrefetchTests(unittest.TestCase):
    def test_detection_continues_while_consumer_is_waiting(self) -> None:
        paths = [Path(f"{index}.jpg") for index in range(5)]
        callbacks: list[Path] = []
        prefetcher = DetectionPrefetcher(
            paths, FakeEngine, threading.Event(), lambda path, _faces: callbacks.append(path), max_buffer=5
        )
        prefetcher.start()
        deadline = time.monotonic() + 2
        while len(prefetcher.buffered_faces()) < 5 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(prefetcher.buffered_faces()), 5)
        self.assertEqual(callbacks, paths)
        self.assertEqual(len(prefetcher.get(paths[0])), 1)
        prefetcher.stop()

    def test_normal_stop_does_not_mark_scan_cancelled(self) -> None:
        cancel = threading.Event()
        prefetcher = DetectionPrefetcher([], FakeEngine, cancel)
        prefetcher.start()
        prefetcher.stop()
        self.assertFalse(cancel.is_set())


if __name__ == "__main__":
    unittest.main()
