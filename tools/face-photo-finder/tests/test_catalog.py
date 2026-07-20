from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from face_finder.catalog import (
    PROFILE_MAX_SAMPLES,
    FaceCatalog,
    KnownIdentity,
    best_known_identity,
    best_unknown_group,
    bounded_profile,
    closest_identity_matches,
    identity_embeddings_for_year,
    image_capture_year,
    image_capture_date,
)


class CatalogTests(unittest.TestCase):
    def test_year_specific_profile_prefers_exact_then_nearest_year(self) -> None:
        old = np.array([1.0, 0.0], dtype=np.float32)
        recent = np.array([0.0, 1.0], dtype=np.float32)
        identity = KnownIdentity(1, "Age Range", (old, recent), (2004, 2025), 1985)
        self.assertEqual(identity_embeddings_for_year(identity, 2004), (old,))
        self.assertEqual(identity_embeddings_for_year(identity, 2023), (recent,))

    def test_capture_year_can_be_read_from_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "college-trip-2004.jpg"
            image.write_bytes(b"not an image")
            self.assertEqual(image_capture_year(image), 2004)

    def test_capture_date_can_be_read_from_camera_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "PXL_20260713_002421467.jpg"
            image.touch()
            self.assertEqual(image_capture_date(image), "2026-07-13")

    def test_gallery_orders_capture_dates_most_recent_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            older = root / "PXL_20260102_120000.jpg"
            newer = root / "PXL_20261230_120000.jpg"
            older.touch()
            newer.touch()
            catalog = FaceCatalog(root / "catalog.sqlite3")
            catalog.store_scan(older, [], [])
            catalog.store_scan(newer, [], [])

            photos = catalog.gallery_photos()

            self.assertEqual([photo["name"] for photo in photos], [newer.name, older.name])
            self.assertEqual([photo["capture_date"] for photo in photos], ["2026-12-30", "2026-01-02"])
            catalog.close()

    def test_birth_year_and_capture_year_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "portrait-2012.jpg"
            image.write_bytes(b"placeholder")
            catalog = FaceCatalog(root / "catalog.sqlite3")
            identity_id = catalog.get_or_create_identity("Dated Person")
            catalog.set_identity_birth_year(identity_id, 1985)
            catalog.store_scan(image, [np.array([1.0, 0.0], dtype=np.float32)], [identity_id])
            identity = catalog.identities()[0]
            self.assertEqual(identity.birth_year, 1985)
            self.assertEqual(identity.sample_years, (2012,))
            self.assertEqual(catalog.cached_image(image).capture_year, 2012)  # type: ignore[union-attr]
            catalog.close()

    def test_identity_management_summaries_and_updates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "identity.jpg"
            image.touch()
            catalog = FaceCatalog(root / "catalog.sqlite3")
            identity_id = catalog.get_or_create_identity("Original Name")
            catalog.store_scan(
                image,
                [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)],
                [identity_id, identity_id],
                profile_eligible=[True, False],
            )

            catalog.update_identity(identity_id, "Updated Name", 1985)
            summary = catalog.identity_summaries()[0]

            self.assertEqual(summary["name"], "Updated Name")
            self.assertEqual(summary["birth_year"], 1985)
            self.assertEqual(summary["photo_count"], 1)
            self.assertEqual(summary["face_count"], 2)
            self.assertEqual(summary["profile_sample_count"], 1)
            with self.assertRaises(ValueError):
                catalog.update_identity(identity_id, "", 1985)
            catalog.close()

    def test_visual_age_and_capture_year_override_persist(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "downloaded-photo.jpg"
            image.write_bytes(b"placeholder")
            catalog = FaceCatalog(root / "catalog.sqlite3")
            identity_id = catalog.get_or_create_identity("Younger Person")
            stored = catalog.store_scan(
                image, [np.array([1.0, 0.0], dtype=np.float32)], [identity_id]
            )
            face_id = catalog.faces_for_image(stored.image_id)[0].face_id
            catalog.set_face_estimated_age(face_id, 17.8)
            self.assertEqual(catalog.set_capture_year_for_faces([face_id], 2006), 1)
            assignment = catalog.identity_assignments(identity_id)[0]
            self.assertAlmostEqual(assignment.estimated_age or 0, 17.8)
            self.assertEqual(assignment.capture_year, 2006)
            self.assertTrue(assignment.capture_year_overridden)
            self.assertEqual(catalog.cached_image(image).capture_year, 2006)  # type: ignore[union-attr]
            catalog.close()

    def test_reconcile_preserves_identity_when_file_moves(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "old" / "portrait.jpg"
            relocated = root / "new" / "portrait.jpg"
            original.parent.mkdir()
            original.write_bytes(b"same-photo-content")
            catalog = FaceCatalog(root / "catalog.sqlite3")
            identity_id = catalog.get_or_create_identity("Moved Person")
            stored = catalog.store_scan(
                original, [np.array([1.0, 0.0], dtype=np.float32)], [identity_id]
            )
            relocated.parent.mkdir()
            original.replace(relocated)
            result = catalog.reconcile_files([relocated])
            self.assertEqual(result.relocated, 1)
            self.assertEqual(result.newly_missing, 0)
            self.assertEqual(catalog.cached_image(relocated).image_id, stored.image_id)  # type: ignore[union-attr]
            assignment = catalog.identity_assignments(identity_id)[0]
            self.assertEqual(assignment.image_path, relocated.resolve())
            self.assertIsNone(assignment.missing_since)
            catalog.close()

    def test_deleted_file_is_marked_missing_and_can_be_pruned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "deleted.jpg"
            image.write_bytes(b"deleted-photo-content")
            catalog = FaceCatalog(root / "catalog.sqlite3")
            identity_id = catalog.get_or_create_identity("Deleted Person")
            catalog.store_scan(image, [np.array([1.0, 0.0], dtype=np.float32)], [identity_id])
            image.unlink()
            result = catalog.reconcile_files([])
            self.assertEqual(result.newly_missing, 1)
            self.assertIsNotNone(catalog.identity_assignments(identity_id)[0].missing_since)
            self.assertEqual(catalog.prune_missing_images(), 1)
            self.assertEqual(catalog.identity_assignments(identity_id), [])
            catalog.close()

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

    def test_delete_photo_removes_source_and_catalog_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "delete-me.jpg"
            image.write_bytes(b"source image")
            catalog = FaceCatalog(root / "catalog.sqlite3")
            stored = catalog.store_scan(
                image, [np.array([1.0, 0.0], dtype=np.float32)], [None]
            )

            deleted = catalog.delete_photo(stored.image_id)

            self.assertEqual(deleted, image)
            self.assertFalse(image.exists())
            self.assertIsNone(catalog.gallery_photo(stored.image_id))
            self.assertEqual(catalog.faces_for_image(stored.image_id), [])
            catalog.close()

    def test_archive_photos_moves_sources_into_zip_and_removes_catalog_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            library = root / "Pictures"
            first = library / "Timeline" / "2025" / "first.jpg"
            second = library / "MISC" / "second.jpg"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.write_bytes(b"first image")
            second.write_bytes(b"second image")
            catalog = FaceCatalog(root / "catalog.sqlite3")
            first_id = catalog.store_scan(first, [], []).image_id
            second_id = catalog.store_scan(second, [], []).image_id
            archive = library / "Hidden Pictures" / "GeneralArchive.zip"

            archived = catalog.archive_photos([first_id, second_id], archive, library)

            self.assertEqual(archived, [first, second])
            self.assertFalse(first.exists())
            self.assertFalse(second.exists())
            self.assertIsNone(catalog.gallery_photo(first_id))
            self.assertIsNone(catalog.gallery_photo(second_id))
            with zipfile.ZipFile(archive) as zipped:
                self.assertEqual(
                    set(zipped.namelist()),
                    {"Timeline/2025/first.jpg", "MISC/second.jpg"},
                )
                self.assertEqual(zipped.read("Timeline/2025/first.jpg"), b"first image")
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

    def test_unknown_group_can_be_retroactively_assigned_to_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_image = root / "first.jpg"
            second_image = root / "second.jpg"
            first_image.write_bytes(b"first image")
            second_image.write_bytes(b"second image")
            catalog = FaceCatalog(root / "catalog.sqlite3")
            group_id = catalog.create_unknown_group()
            first = catalog.store_scan(
                first_image,
                [np.array([1.0, 0.0], dtype=np.float32)],
                [None],
                intentionally_unknown=[True],
                unknown_group_ids=[group_id],
            )
            second = catalog.store_scan(
                second_image,
                [np.array([0.98, 0.02], dtype=np.float32)],
                [None],
                intentionally_unknown=[True],
                unknown_group_ids=[group_id],
            )
            self.assertEqual(catalog.unknown_group_face_count(group_id), 2)

            identity_id = catalog.get_or_create_identity("Now Known")
            self.assertEqual(catalog.assign_unknown_group(group_id, identity_id), (2, 2))
            for stored in (first, second):
                face = catalog.faces_for_image(stored.image_id)[0]
                self.assertEqual(face.identity_name, "Now Known")
                self.assertFalse(face.intentionally_unknown)
                self.assertIsNone(face.unknown_group_id)
                self.assertEqual(catalog.cached_image(stored.path).identified_count, 1)  # type: ignore[union-attr]
            self.assertNotIn(group_id, [group.group_id for group in catalog.unknown_groups()])
            catalog.close()

    def test_gallery_metadata_can_be_saved_and_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "holiday.jpg"
            image.write_bytes(b"gallery image")
            catalog = FaceCatalog(root / "catalog.sqlite3")
            stored = catalog.store_scan(image, [], [])
            catalog.set_nsfw_classification(
                stored.image_id,
                0.82,
                [{"label": "FEMALE_BREAST_EXPOSED", "score": 0.82, "explicit": True}],
                0.91,
                True,
            )
            catalog.update_gallery_metadata(
                stored.image_id, "Summer trip", "At the lake", "family, vacation", 5, 2018, 1,
                "document", "Lake Michigan", 43.0, -87.0,
            )
            photo = catalog.gallery_photo(stored.image_id)
            self.assertIsNotNone(photo)
            self.assertEqual(photo["title"], "Summer trip")  # type: ignore[index]
            self.assertEqual(photo["year"], 2018)  # type: ignore[index]
            self.assertEqual(photo["rating"], 5)  # type: ignore[index]
            self.assertTrue(photo["is_nsfw"])  # type: ignore[index]
            self.assertEqual(photo["media_kind"], "document")  # type: ignore[index]
            self.assertEqual(photo["location_name"], "Lake Michigan")  # type: ignore[index]
            self.assertEqual(photo["latitude"], 43.0)  # type: ignore[index]
            self.assertEqual(photo["longitude"], -87.0)  # type: ignore[index]
            self.assertEqual(photo["nsfw_detections"][0]["label"], "FEMALE_BREAST_EXPOSED")  # type: ignore[index]
            self.assertEqual(photo["nsfw_vit_score"], 0.91)  # type: ignore[index]
            self.assertTrue(photo["nsfw_review_required"])  # type: ignore[index]
            self.assertEqual([item["id"] for item in catalog.gallery_photos(search="vacation")], [stored.image_id])
            self.assertEqual([item["id"] for item in catalog.gallery_photos(year=2018)], [stored.image_id])
            self.assertEqual([item["id"] for item in catalog.gallery_photos(nsfw_filter="nsfw")], [stored.image_id])
            self.assertEqual(catalog.gallery_photos(nsfw_filter="safe"), [])
            self.assertEqual(catalog.gallery_photos(nsfw_filter="review"), [])
            self.assertEqual(catalog.gallery_photos(nsfw_filter="conflict"), [])
            self.assertEqual(catalog.set_nsfw_overrides([stored.image_id], None), 1)
            self.assertEqual(
                [item["id"] for item in catalog.gallery_photos(nsfw_filter="conflict")],
                [stored.image_id],
            )
            self.assertEqual(catalog.set_nsfw_overrides([stored.image_id], 0), 1)
            self.assertEqual(catalog.gallery_photos(nsfw_filter="conflict"), [])
            self.assertEqual(catalog.gallery_photos(excluded_kinds=("document",)), [])
            self.assertEqual(
                [item["id"] for item in catalog.gallery_photos(media_kind="document")],
                [stored.image_id],
            )
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
