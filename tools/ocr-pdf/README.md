# OCR PDF

Create a searchable PDF from either an image-based PDF or an ordered collection of image files. All OCR happens locally on the computer.

## What it does

- adds a searchable text layer to scanned or image-only PDFs;
- combines JPG, PNG, TIFF, WebP, BMP, or GIF files into one PDF before OCR;
- keeps supplied image files in the supplied order; and
- sorts a folder of images by filename, making names such as `page-001.jpg`, `page-002.jpg`, and so on work naturally.

The original files are never modified.

## Requirements

- Python 3.10 or newer;
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki), installed and available on `PATH`;
- [Ghostscript](https://www.ghostscript.com/releases/gsdnld.html), installed and available on `PATH`; and
- the Python dependencies below.

Set up an isolated environment from this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For languages other than English, install the matching Tesseract language data first. Use `--language eng+spa` for documents containing both English and Spanish, for example.

## Usage

```powershell
# Convert an existing scanned PDF.
python ocr_pdf.py "C:\Scans\contract.pdf" "C:\Scans\contract-searchable.pdf"

# Combine image files in the specified order, then OCR them.
python ocr_pdf.py "C:\Scans\page-01.jpg" "C:\Scans\page-02.jpg" "C:\Scans\page-03.jpg" "C:\Scans\contract-searchable.pdf"

# Convert every supported image in a folder, sorted by filename.
python ocr_pdf.py "C:\Scans\document-pages" "C:\Scans\document-searchable.pdf"
```

By default the tool preserves any existing text in a PDF and straightens slightly tilted pages. Add `--force-ocr` to OCR every page, or `--no-deskew` to leave page rotation untouched.

## Limitations and safety

OCR accuracy depends on scan quality, handwriting, language data, and page layout. Review important names, amounts, signatures, and dates against the original document. Password-protected PDFs and damaged image files may need to be repaired or unlocked before conversion.

The output contains a text layer and may be larger than the source document. The tool does not upload documents or send them to an external service.

## Tests

```powershell
python -m unittest discover -s tests -v
```
