from pathlib import Path
import tempfile
import unittest

import numpy as np

from face_finder.catalog import (
    PROFILE_MAX_SAMPLES,
    FaceCatalog,
    KnownIdentity,
    best_known_identity,
    best_unknown_group,
    bounded_profile,
    closest_identity_matches,
)


class CatalogTests(unittest.TestCase):
    def test_closest_identity_matches_are_ranked_with_percent_ready_scores(self) -> None:
        identities = [
            KnownIdentity(1, "Second", (np.array([0.7, 0.3], dtype=np.float32),)),
            KnownIdentity(2, "Closest", (np.array([0.9, 0.1], dtype=np.float32),)),
            KnownIdentity(3, "No Samples", ()),
        ]
        matches = closest_identity_matches(np.array([1.0, 0.0], dtype=np.float32), identities)
        self.assertEqual([item.name for item, _score in matches], ["Closest", "Second"])
        self.assertAlmostEqual(matches[0][1], 0.9, places=5)

    def test_face_assignments_can_be_moved_and_empty_duplicate_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "duplicate-profile.jpg"
            image.touch()
            catalog = FaceCatalog(root / "catalog.sqlite3")
            duplicate_id = catalog.get_or_create_identity("Alex Smit")
            correct_id = catalog.get_or_create_identity("Alex Smith")
            stored = catalog.store_scan(
                image,
                [np.array([1.0, 0.0], dtype=np.float32)],
                [duplicate_id],
                profile_eligible=[False],
                is_art=[True],
            )
            assignment = catalog.identity_assignments(duplicate_id)[0]
            self.assertEqual(assignment.image_path, image)
            self.assertTrue(assignment.is_art)
            self.assertEqual(catalog.reassign_faces([assignment.face_id], correct_id), 1)
            self.assertTrue(catalog.remove_identity_if_unused(duplicate_id))
            moved = catalog.faces_for_image(stored.image_id)[0]
            self.assertEqual(moved.identity_name, "Alex Smith")
            self.assertTrue(moved.is_art)
            self.assertFalse(moved.profile_eligible)
            self.assertNotIn("Alex Smit", [item.name for item in catalog.identities()])
            catalog.close()

    def test_art_face_is_linked_to_identity_but_excluded_from_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "portrait-drawing.jpg"
            image.touch()
            catalog = FaceCatalog(root / "catalog.sqlite3")
            person_id = catalog.get_or_create_identity("Artwork Subject")
            stored = catalog.store_scan(
                image,
                [np.array([1.0, 0.0], dtype=np.float32)],
                [person_id],
                profile_eligible=[False],
                is_art=[True],
            )
            face = catalog.faces_for_image(stored.image_id)[0]
            self.assertEqual(face.identity_name, "Artwork Subject")
            self.assertTrue(face.is_art)
            self.assertFalse(face.profile_eligible)
            identity = next(item for item in catalog.identities() if item.identity_id == person_id)
            self.assertEqual(identity.embeddings, ())
            catalog.close()

    def test_reset_removes_all_catalog_data_without_deleting_source_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "source.jpg"
            image.write_bytes(b"source image")
            catalog = FaceCatalog(root / "catalog.sqlite3")
            person_id = catalog.get_or_create_identity("Reset Person")
            catalog.store_scan(image, [np.array([1.0, 0.0], dtype=np.float32)], [person_id])
            catalog.create_unknown_group()
            catalog.reset()
            self.assertEqual(catalog.identities(), [])
            self.assertEqual(catalog.unknown_groups(), [])
            self.assertIsNone(catalog.cached_image(image))
            self.assertTrue(image.exists())
            catalog.close()

    def test_blurry_assigned_face_is_not_used_as_profile_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "blurry.jpg"
            image.touch()
            catalog = FaceCatalog(root / "catalog.sqlite3")
            person_id = catalog.get_or_create_identity("Blurry Person")
            stored = catalog.store_scan(
                image,
                [np.array([1.0, 0.0], dtype=np.float32)],
                [person_id],
                sharpness_scores=[12.0],
                profile_eligible=[False],
            )
            face = catalog.faces_for_image(stored.image_id)[0]
            self.assertEqual(face.identity_name, "Blurry Person")
            self.assertFalse(face.profile_eligible)
            identity = next(item for item in catalog.identities() if item.identity_id == person_id)
            self.assertEqual(identity.embeddings, ())
            catalog.close()

    def test_profile_drops_near_duplicates_and_caps_growth(self) -> None:
        duplicate = np.array([1.0, 0.0], dtype=np.float32)
        samples = [duplicate, duplicate.copy()]
        samples.extend(
            np.array([np.cos(index), np.sin(index)], dtype=np.float32)
            for index in np.linspace(0.1, 6.0, PROFILE_MAX_SAMPLES * 3)
        )
        profile = bounded_profile(samples)
        self.assertLessEqual(len(profile), PROFILE_MAX_SAMPLES)
        self.assertEqual(sum(np.array_equal(item, duplicate) for item in profile), 1)

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
            group_id = catalog.create_unknown_group()
            stored = catalog.store_scan(
                image,
                [np.array([1.0, 0.0], dtype=np.float32)],
                [None],
                intentionally_unknown=[True],
                unknown_group_ids=[group_id],
            )
            face = catalog.faces_for_image(stored.image_id)[0]
            self.assertTrue(face.intentionally_unknown)
            self.assertEqual(face.unknown_group_id, group_id)
            matched_group, _score = best_unknown_group(
                np.array([0.98, 0.02], dtype=np.float32), catalog.unknown_groups(), 0.8
            )
            self.assertEqual(matched_group, group_id)
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
