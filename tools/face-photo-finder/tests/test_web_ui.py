import unittest

from face_finder.web_server import PAGE


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
        self.assertIn('contributes to NSFW score', PAGE)
        self.assertNotIn('>Save metadata</button>', PAGE)


if __name__ == "__main__":
    unittest.main()
