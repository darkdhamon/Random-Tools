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
    fallback_identity_embeddings_for_year,
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

    def test_blurry_samples_are_used_only_after_clear_samples_fail(self) -> None:
        clear = np.array([1.0, 0.0], dtype=np.float32)
        blurry = np.array([0.0, 1.0], dtype=np.float32)
        clear_winner = KnownIdentity(1, "Clear winner", (clear,))
        fallback_winner = KnownIdentity(2, "Fallback winner", (), (), None, (blurry,), (2020,))

        identity_id, _score = best_known_identity(
            np.array([0.8, 0.95], dtype=np.float32), [clear_winner, fallback_winner], 0.75
        )
        self.assertEqual(identity_id, 1)

        identity_id, _score = best_known_identity(
            np.array([0.2, 0.95], dtype=np.float32), [clear_winner, fallback_winner], 0.75
        )
        self.assertEqual(identity_id, 2)
        self.assertEqual(fallback_identity_embeddings_for_year(fallback_winner, 2020), (blurry,))

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
                previews=[
                    np.frombuffer(b"\xff\xd8reference-one", dtype=np.uint8),
                    np.frombuffer(b"\xff\xd8reference-two", dtype=np.uint8),
                ],
                profile_eligible=[True, False],
            )

            catalog.update_identity(identity_id, "Updated Name", 1985)
            summary = catalog.identity_summaries()[0]

            self.assertEqual(summary["name"], "Updated Name")
            self.assertEqual(summary["birth_year"], 1985)
            self.assertEqual(summary["photo_count"], 1)
            self.assertEqual(summary["face_count"], 2)
            self.assertEqual(summary["profile_sample_count"], 1)
            self.assertEqual(len(summary["reference_face_ids"]), 1)
            self.assertEqual(
                catalog.identity_reference_preview(summary["reference_face_ids"][0]),
                b"\xff\xd8reference-one",
            )
            with self.assertRaises(ValueError):
                catalog.update_identity(identity_id, "", 1985)
            catalog.close()

    def test_identity_reference_images_are_newest_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            older = root / "PXL_20200102_120000.jpg"; older.write_bytes(b"older")
            newer = root / "PXL_20251231_120000.jpg"; newer.write_bytes(b"newer")
            catalog = FaceCatalog(root / "catalog.sqlite3")
            identity_id = catalog.get_or_create_identity("Dated references")
            old_scan = catalog.store_scan(
                older, [np.array([1.0, 0.0], dtype=np.float32)], [identity_id],
                previews=[np.frombuffer(b"old-preview", dtype=np.uint8)],
            )
            new_scan = catalog.store_scan(
                newer, [np.array([0.0, 1.0], dtype=np.float32)], [identity_id],
                previews=[np.frombuffer(b"new-preview", dtype=np.uint8)],
            )
            old_face_id = catalog.faces_for_image(old_scan.image_id)[0].face_id
            new_face_id = catalog.faces_for_image(new_scan.image_id)[0].face_id
            self.assertEqual(
                catalog.identity_summaries()[0]["reference_face_ids"],
                [new_face_id, old_face_id],
            )
            self.assertEqual(catalog.identity_summaries()[0]["last_seen"], "2025-12-31")
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

    def test_face_assignment_suggestions_prioritize_same_day_then_match(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = FaceCatalog(root / "catalog.sqlite3")
            same_day_id = catalog.get_or_create_identity("Same Day")
            closest_id = catalog.get_or_create_identity("Closest Overall")
            empty_id = catalog.create_identity("No Samples")
            same_day = root / "PXL_20260713_010000_same.jpg"; same_day.touch()
            other_day = root / "PXL_20260712_010000_other.jpg"; other_day.touch()
            target = root / "PXL_20260713_020000_target.jpg"; target.touch()
            catalog.store_scan(same_day, [np.array([0.6, 0.8], dtype=np.float32)], [same_day_id])
            catalog.store_scan(other_day, [np.array([0.99, 0.1], dtype=np.float32)], [closest_id])
            stored = catalog.store_scan(target, [np.array([1.0, 0.0], dtype=np.float32)], [None])

            suggestions = catalog.face_assignment_suggestions(
                catalog.faces_for_image(stored.image_id)[0].face_id
            )

            self.assertEqual([item["id"] for item in suggestions], [same_day_id, closest_id, empty_id])
            self.assertTrue(suggestions[0]["same_day"])
            self.assertGreater(suggestions[1]["match_score"], suggestions[0]["match_score"])
            self.assertIsNone(suggestions[2]["match_score"])
            catalog.close()

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

    def test_duplicate_names_have_distinct_ids_and_can_be_merged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_image = root / "first.jpg"; first_image.write_bytes(b"first")
            second_image = root / "second.jpg"; second_image.write_bytes(b"second")
            catalog_path = root / "catalog.sqlite3"
            catalog = FaceCatalog(catalog_path)
            keep_id = catalog.create_identity("Same Name", 1980)
            merge_id = catalog.create_identity("Same Name", 1990)
            self.assertNotEqual(keep_id, merge_id)
            first = catalog.store_scan(first_image, [np.array([1.0, 0.0], dtype=np.float32)], [keep_id])
            second = catalog.store_scan(second_image, [np.array([0.0, 1.0], dtype=np.float32)], [merge_id])
            catalog.add_photo_identity_tag(first.image_id, keep_id)
            catalog.add_photo_identity_tag(first.image_id, merge_id)

            moved_faces, moved_tags = catalog.merge_identities([keep_id, merge_id], keep_id)
            self.assertEqual(moved_faces, 1)
            self.assertEqual(moved_tags, 1)
            identities = catalog.identities()
            self.assertEqual([(item.identity_id, item.name, item.birth_year) for item in identities], [(keep_id, "Same Name", 1980)])
            self.assertEqual(catalog.faces_for_image(second.image_id)[0].identity_id, keep_id)
            self.assertEqual(catalog.gallery_photo(first.image_id)["face_tags"], [{"identity_id": keep_id, "name": "Same Name", "target_x": None, "target_y": None}])  # type: ignore[index]
            catalog.close()

            reopened = FaceCatalog(catalog_path)
            another_id = reopened.create_identity("Same Name")
            self.assertNotEqual(another_id, keep_id)
            reopened.close()

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
            self.assertEqual(identity.fallback_embeddings, ())
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
            self.assertEqual(len(identity.fallback_embeddings), 1)
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
            self.assertEqual(catalog.gallery_photo(stored.image_id)["faces"][0]["bbox"], [10, 20, 30, 40])  # type: ignore[index]
            self.assertIsNone(faces[1].identity_name)
            catalog.assign_face(faces[1].face_id, person_id)
            self.assertEqual(catalog.cached_image(image).identified_count, 2)  # type: ignore[union-attr]
            catalog.remove_face(faces[1].face_id)
            after_removal = catalog.cached_image(image)
            self.assertEqual(after_removal.face_count, 1)  # type: ignore[union-attr]
            self.assertEqual(after_removal.identified_count, 1)  # type: ignore[union-attr]
            catalog.close()

    def test_manual_person_tags_are_non_biometric_and_searchable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "car.jpg"
            image.write_bytes(b"image placeholder")
            catalog = FaceCatalog(root / "catalog.sqlite3")
            stored = catalog.store_scan(image, [], [])
            person_id = catalog.get_or_create_identity("Farv")

            catalog.add_photo_identity_tag(stored.image_id, person_id, 0.25, 0.75)
            photo = catalog.gallery_photo(stored.image_id)
            self.assertEqual(photo["face_count"], 0)  # type: ignore[index]
            self.assertEqual(
                photo["face_tags"],  # type: ignore[index]
                [{"identity_id": person_id, "name": "Farv", "target_x": 0.25, "target_y": 0.75}],
            )
            matches = catalog.gallery_photos(identity_id=person_id)
            self.assertEqual([item["id"] for item in matches], [stored.image_id])
            identity = next(item for item in catalog.identities() if item.identity_id == person_id)
            self.assertEqual(identity.embeddings, ())

            catalog.add_photo_identity_tag(stored.image_id, person_id, 0.5, 0.4)
            retargeted = catalog.gallery_photo(stored.image_id)["face_tags"][0]  # type: ignore[index]
            self.assertEqual((retargeted["target_x"], retargeted["target_y"]), (0.5, 0.4))
            with self.assertRaises(ValueError):
                catalog.add_photo_identity_tag(stored.image_id, person_id, 1.1, 0.5)

            catalog.remove_photo_identity_tag(stored.image_id, person_id)
            self.assertEqual(catalog.gallery_photo(stored.image_id)["face_tags"], [])  # type: ignore[index]
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

    def test_removing_false_unknown_reference_cleans_up_empty_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "false-face.jpg"; image.write_bytes(b"false")
            catalog = FaceCatalog(root / "catalog.sqlite3")
            group_id = catalog.create_unknown_group()
            stored = catalog.store_scan(
                image, [np.array([1.0, 0.0], dtype=np.float32)], [None],
                intentionally_unknown=[True], unknown_group_ids=[group_id],
                previews=[np.frombuffer(b"preview", dtype=np.uint8)],
            )
            face_id = catalog.faces_for_image(stored.image_id)[0].face_id
            catalog.remove_face(face_id)
            self.assertEqual(catalog.cached_image(image).face_count, 0)  # type: ignore[union-attr]
            self.assertEqual(catalog.unidentified_summaries(), [])
            self.assertEqual(catalog.unknown_groups(), [])
            catalog.close()

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
            summaries = catalog.unidentified_summaries()
            self.assertEqual(summaries[0]["id"], group_id)
            self.assertEqual(summaries[0]["face_count"], 1)
            self.assertEqual(summaries[0]["photo_count"], 1)
            self.assertEqual(
                [item["id"] for item in catalog.gallery_photos(unknown_group_id=group_id)],
                [stored.image_id],
            )
            person_id = catalog.get_or_create_identity("Later Identified")
            catalog.assign_face(face.face_id, person_id)
            identified = catalog.faces_for_image(stored.image_id)[0]
            self.assertFalse(identified.intentionally_unknown)
            self.assertEqual(identified.identity_name, "Later Identified")
            self.assertEqual(catalog.unidentified_summaries(), [])
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

    def test_similar_unknown_groups_are_clustered_and_assigned_together(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog = FaceCatalog(root / "catalog.sqlite3")
            group_ids = [catalog.create_unknown_group() for _ in range(3)]
            embeddings = [
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([0.99, 0.01], dtype=np.float32),
                np.array([0.0, 1.0], dtype=np.float32),
            ]
            capture_dates = ["20200101", "20211231", "20220615"]
            image_ids = []
            for index, (group_id, embedding) in enumerate(zip(group_ids, embeddings, strict=True)):
                image = root / f"PXL_{capture_dates[index]}_120000_unknown-{index}.jpg"; image.write_bytes(bytes([index + 1]))
                stored = catalog.store_scan(
                    image, [embedding], [None], intentionally_unknown=[True],
                    unknown_group_ids=[group_id],
                    previews=[np.frombuffer(f"preview-{index}".encode(), dtype=np.uint8)],
                )
                image_ids.append(stored.image_id)

            summaries = catalog.unidentified_summaries()
            self.assertEqual(len(summaries), 2)
            similar = next(item for item in summaries if len(item["group_ids"]) == 2)
            self.assertEqual(set(similar["group_ids"]), set(group_ids[:2]))
            self.assertEqual(similar["face_count"], 2)
            self.assertEqual(similar["last_seen"], "2021-12-31")
            self.assertEqual(
                {item["id"] for item in catalog.gallery_photos(unknown_group_ids=tuple(group_ids[:2]))},
                set(image_ids[:2]),
            )

            person_id = catalog.create_identity("Now recognized")
            self.assertEqual(catalog.assign_unknown_groups(similar["group_ids"], person_id), (2, 2))
            self.assertEqual(
                {catalog.faces_for_image(image_id)[0].identity_id for image_id in image_ids[:2]},
                {person_id},
            )
            self.assertEqual(len(catalog.unidentified_summaries()), 1)
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
