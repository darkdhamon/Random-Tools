from pathlib import Path
import tempfile
import unittest

import numpy as np

from face_finder.catalog import FaceCatalog, best_known_identity


class CatalogTests(unittest.TestCase):
    def test_scan_cache_identity_and_image_stats_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "photo.jpg"
            image.write_bytes(b"image placeholder")
            catalog = FaceCatalog(root / "catalog.sqlite3")
            person_id = catalog.get_or_create_identity("Alex Example")
            first = np.array([1.0, 0.0], dtype=np.float32)
            second = np.array([0.0, 1.0], dtype=np.float32)
            stored = catalog.store_scan(
                image, [first, second], [person_id, None], boxes=[(10, 20, 30, 40), (50, 60, 70, 80)]
            )
            cached = catalog.cached_image(image)
            self.assertEqual(cached, stored)
            self.assertEqual(cached.face_count, 2)  # type: ignore[union-attr]
            self.assertEqual(cached.identified_count, 1)  # type: ignore[union-attr]
            faces = catalog.faces_for_image(stored.image_id)
            self.assertEqual(faces[0].identity_name, "Alex Example")
            self.assertEqual(faces[0].bbox, (10, 20, 30, 40))
            self.assertIsNone(faces[1].identity_name)
            catalog.assign_face(faces[1].face_id, person_id)
            self.assertEqual(catalog.cached_image(image).identified_count, 2)  # type: ignore[union-attr]
            catalog.remove_face(faces[1].face_id)
            after_removal = catalog.cached_image(image)
            self.assertEqual(after_removal.face_count, 1)  # type: ignore[union-attr]
            self.assertEqual(after_removal.identified_count, 1)  # type: ignore[union-attr]
            catalog.close()

    def test_modified_image_is_not_returned_as_cached(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "photo.jpg"
            image.write_bytes(b"first")
            catalog = FaceCatalog(root / "catalog.sqlite3")
            catalog.store_scan(image, [], [])
            image.write_bytes(b"different size")
            self.assertIsNone(catalog.cached_image(image))
            catalog.close()

    def test_unidentified_faces_remain_available_after_reopening_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "group.jpg"
            image.touch()
            catalog_path = root / "catalog.sqlite3"
            catalog = FaceCatalog(catalog_path)
            stored = catalog.store_scan(
                image,
                [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)],
                [None, None],
            )
            catalog.close()
            reopened = FaceCatalog(catalog_path)
            faces = reopened.faces_for_image(stored.image_id)
            self.assertEqual(len(faces), 2)
            self.assertTrue(all(face.identity_id is None for face in faces))
            reopened.close()

    def test_intentionally_unknown_face_is_persisted_and_can_later_be_identified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "public-event.jpg"
            image.touch()
            catalog = FaceCatalog(root / "catalog.sqlite3")
            stored = catalog.store_scan(
                image, [np.array([1.0, 0.0], dtype=np.float32)], [None], intentionally_unknown=[True]
            )
            face = catalog.faces_for_image(stored.image_id)[0]
            self.assertTrue(face.intentionally_unknown)
            person_id = catalog.get_or_create_identity("Later Identified")
            catalog.assign_face(face.face_id, person_id)
            identified = catalog.faces_for_image(stored.image_id)[0]
            self.assertFalse(identified.intentionally_unknown)
            self.assertEqual(identified.identity_name, "Later Identified")
            catalog.close()

    def test_best_known_identity_applies_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            catalog = FaceCatalog(Path(temporary) / "catalog.sqlite3")
            person_id = catalog.get_or_create_identity("Morgan")
            photo = Path(temporary) / "person.jpg"
            photo.touch()
            catalog.store_scan(photo, [np.array([1.0, 0.0], dtype=np.float32)], [person_id])
            identities = catalog.identities()
            self.assertEqual(best_known_identity(np.array([0.95, 0.05]), identities, 0.8)[0], person_id)
            self.assertIsNone(best_known_identity(np.array([0.4, 0.6]), identities, 0.8)[0])
            catalog.close()


if __name__ == "__main__":
    unittest.main()
