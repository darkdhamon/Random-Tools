# Face Photo Finder

A private, local desktop tool that recursively scans a folder and finds photos likely to contain the person shown in one or more reference photos.

## Features

- scans the selected folder and every subfolder;
- supports JPG, JPEG, PNG, WebP, BMP, TIFF, and TIF;
- uses multiple reference photos for better coverage;
- shows every match with its similarity score;
- opens matches, exports a CSV report, or copies selected/all matches;
- never modifies source photos and does not upload images or face data.

## Requirements and setup

- Windows, macOS, or Linux with Python 3.10 or newer (Python 3.11–3.13 recommended)
- internet access on the first run to download two OpenCV Zoo model files (about 37 MB total)

```powershell
cd "tools/face-photo-finder"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run.py
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

## Use

1. Select one or more clear reference photos. Each reference must contain exactly one face.
2. Select the folder containing the photo collection.
3. Leave match strictness at `0.45` initially and start the scan.
4. Review the ranked results. Double-click a row to open the original.
5. Export paths as CSV or copy matches to another folder. Copying preserves originals and resolves duplicate filenames automatically.

A lower threshold finds more possible matches but creates more false positives. A higher threshold is stricter but may miss the person. Different ages, angles, lighting, glasses, masks, and small or blurry faces affect accuracy.

## Privacy, consent, and limitations

Face embeddings are biometric data. Use this tool only on photos you are authorized to process and follow applicable consent, privacy, and retention requirements. Processing stays on the computer, but the first run downloads the MIT-licensed YuNet and SFace models from the official OpenCV Zoo.

Face recognition is probabilistic and can perform differently across demographic groups. Results are leads, not proof of identity. Review every result manually; do not use the tool for high-impact decisions, surveillance, or identification without consent.

## Tests

```powershell
python -m unittest discover -s tests -v
```

