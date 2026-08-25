from pathlib import Path
import tempfile
import unittest

from PIL import Image

from face_finder.web_server import media_bytes


class ThumbnailCacheTests(unittest.TestCase):
    def test_thumbnail_is_cached_and_source_changes_invalidate_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "photo.jpg"
            cache = root / "cache"
            Image.new("RGB", (900, 600), "red").save(source)

            first, first_hit = media_bytes(source, 7, True, False, cache)
            second, second_hit = media_bytes(source, 7, True, False, cache)
            self.assertFalse(first_hit)
            self.assertTrue(second_hit)
            self.assertEqual(first, second)

            Image.new("RGB", (900, 600), "blue").save(source)
            changed, changed_hit = media_bytes(source, 7, True, False, cache)
            self.assertFalse(changed_hit)
            self.assertNotEqual(first, changed)

    def test_blurred_and_normal_thumbnails_have_separate_cache_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "photo.jpg"
            cache = root / "cache"
            Image.new("RGB", (900, 600), "red").save(source)

            media_bytes(source, 9, True, False, cache)
            _, blurred_hit = media_bytes(source, 9, True, True, cache)
            self.assertFalse(blurred_hit)
            self.assertEqual(len(list(cache.rglob("*.jpg"))), 2)
            self.assertTrue(media_bytes(source, 9, True, True, cache)[1])
