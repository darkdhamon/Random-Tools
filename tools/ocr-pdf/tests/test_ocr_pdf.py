import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image


MODULE_PATH = Path(__file__).parents[1] / "ocr_pdf.py"
SPEC = importlib.util.spec_from_file_location("ocr_pdf", MODULE_PATH)
ocr_pdf = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ocr_pdf)


class CreateImagePdfTests(unittest.TestCase):
    def test_combines_images_in_order_into_a_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first = directory / "page-01.png"
            second = directory / "page-02.png"
            output = directory / "combined.pdf"
            Image.new("RGB", (20, 10), "red").save(first)
            Image.new("RGB", (30, 10), "blue").save(second)

            ocr_pdf.create_image_pdf([first, second], output)

            self.assertTrue(output.exists())
            self.assertEqual(output.read_bytes()[:5], b"%PDF-")

    def test_collect_images_sorts_a_folder_by_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            for name in ("page-10.jpg", "page-02.jpg", "page-01.jpg"):
                Image.new("RGB", (5, 5), "white").save(directory / name)

            images = ocr_pdf.collect_images([directory])

            self.assertEqual([image.name for image in images], ["page-01.jpg", "page-02.jpg", "page-10.jpg"])


if __name__ == "__main__":
    unittest.main()
