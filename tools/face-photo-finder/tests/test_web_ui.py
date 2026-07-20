import unittest

from face_finder.web_server import PAGE, GENERAL_ARCHIVE, archive_path_for_name


class WebGalleryUiTests(unittest.TestCase):
    def test_gallery_uses_timeline_and_automatic_metadata_saving(self) -> None:
        self.assertIn('id=timeline class=timeline', PAGE)
        self.assertIn('className=\'year-group\'', PAGE)
        self.assertIn('setTimeout(save,immediate?0:700)', PAGE)
        self.assertIn('Saved automatically', PAGE)
        self.assertIn('id=map class=map', PAGE)
        self.assertIn('Open in OpenStreetMap', PAGE)
        self.assertIn("'locationName','latitude','longitude'", PAGE)
        self.assertIn('Model detections', PAGE)
        self.assertIn('automatic NSFW decision', PAGE)
        self.assertIn('explicit-content evidence', PAGE)
        self.assertIn('NSFW conflicts', PAGE)
        self.assertIn('Models disagree', PAGE)
        self.assertIn('Danger: permanently delete photo?', PAGE)
        self.assertIn('This action cannot be undone.', PAGE)
        self.assertIn("post('/api/delete-photo'", PAGE)
        self.assertIn("new IntersectionObserver", PAGE)
        self.assertIn("rootMargin:'600px 0px'", PAGE)
        self.assertIn("if(loading){if(reset)reloadAfterLoad=true;return}", PAGE)
        self.assertIn("function timelineGrid(photo)", PAGE)
        self.assertIn("className='month-group'", PAGE)
        self.assertIn("className='day-group'", PAGE)
        self.assertIn("weekday:'long',month:'long',day:'numeric'", PAGE)
        self.assertIn("c.dataset.photoId=x.id", PAGE)
        self.assertIn("removeDeletedCard(deletedId)", PAGE)
        self.assertIn("offset = Math.max(0, offset - 1)", PAGE)
        self.assertIn("deleteScrollPosition = window.scrollY", PAGE)
        self.assertIn("requestAnimationFrame(() => window.scrollTo", PAGE)
        self.assertIn("Select photos", PAGE)
        self.assertIn("Archive selected", PAGE)
        self.assertIn("Delete selected", PAGE)
        self.assertIn("const selectedPhotos = new Map()", PAGE)
        self.assertIn("'/api/archive-photos' : '/api/delete-photos'", PAGE)
        self.assertIn("Existing archive", PAGE)
        self.assertIn("Or create a new archive", PAGE)
        self.assertIn("payload.archive_name", PAGE)
        self.assertNotIn("current = null;\n    await load();", PAGE)
        self.assertNotIn('>Save metadata</button>', PAGE)

    def test_archive_destination_defaults_and_rejects_paths(self) -> None:
        self.assertEqual(archive_path_for_name(), GENERAL_ARCHIVE)
        self.assertEqual(archive_path_for_name("Trips"), GENERAL_ARCHIVE.parent / "Trips.zip")
        self.assertEqual(
            archive_path_for_name("Existing.hide"), GENERAL_ARCHIVE.parent / "Existing.hide"
        )
        with self.assertRaises(ValueError):
            archive_path_for_name("../outside.zip")
        with self.assertRaises(ValueError):
            archive_path_for_name("bad.exe")


if __name__ == "__main__":
    unittest.main()
