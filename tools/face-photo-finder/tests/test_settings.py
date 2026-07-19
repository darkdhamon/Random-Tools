from pathlib import Path
import tempfile
import unittest

from face_finder.settings import AppSettings, load_settings, save_settings


class SettingsTests(unittest.TestCase):
    def test_round_trip_existing_reference_and_folder_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "person.jpg"
            reference.touch()
            settings_file = root / "preferences" / "settings.json"
            expected = AppSettings((reference,), root)
            save_settings(expected, settings_file)
            self.assertEqual(load_settings(settings_file), expected)

    def test_missing_paths_are_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings_file = root / "settings.json"
            save_settings(AppSettings((root / "missing.jpg",), root / "missing-folder"), settings_file)
            self.assertEqual(load_settings(settings_file), AppSettings())

    def test_invalid_json_returns_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            settings_file = Path(temporary) / "settings.json"
            settings_file.write_text("not json", encoding="utf-8")
            self.assertEqual(load_settings(settings_file), AppSettings())


if __name__ == "__main__":
    unittest.main()

