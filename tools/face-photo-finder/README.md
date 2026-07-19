# Face Photo Finder

A private, local desktop tool that recursively scans a folder and finds photos likely to contain the person shown in one or more reference photos.

## Features

- scans the selected folder and every subfolder;
- supports JPG, JPEG, PNG, WebP, BMP, TIFF, and TIF;
- skips files whose filename contains `screenshot` (case-insensitive);
- automatically scales high-resolution phone photos for reliable close-up face detection while recognizing against the original image;
- uses multiple reference photos for better coverage;
- shows a face picker when a reference photo contains multiple people, allowing one or several faces to be selected;
- shows every match with its similarity score;
- displays oriented thumbnails for the reference images and every matching photo;
- toggles results between a detailed list and a large-thumbnail gallery with selectable cards;
- restores the previously selected reference photos and last-used search folder when reopened;
- maintains a local SQLite face catalog with identity names, embeddings, image fingerprints, detected-face counts, and identified-face counts;
- reuses cached results for unchanged photos and analyzes only new or modified files;
- pre-processes upcoming uncached photos with a separate bounded background detector while identity questions are open;
- catalogs all face-containing photos when no reference or known person is selected, prompting to name unfamiliar faces;
- searches by a previously learned person from the **Known person** list without requiring another reference photo;
- autocompletes identity names case-insensitively while still allowing new names;
- shows up to three closest named-profile buttons with match percentages when automatic recognition is uncertain;
- provides a two-step danger-confirmed **Reset face database** action—including typing `RESET`—that removes all biometric and scan-cache data without touching source photos or saved preferences;
- includes **Manage identities** for reviewing face/photo assignments, moving selected entries, or merging an entire duplicate profile into an existing or newly named profile;
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

1. To find a specific person, select clear reference photos or choose someone from **Known person**. If a reference contains multiple faces, choose one or several people from the thumbnail picker.
2. Select the folder containing the photo collection.
3. Leave match strictness at `0.45` initially and start the scan.
4. Review the ranked results. Double-click a row to open the original.
5. Export paths as CSV or copy matches to another folder. Copying preserves originals and resolves duplicate filenames automatically.

To build or update the identity catalog, clear the reference photos, leave **Known person** blank, and start a scan. Every photo containing a face is included, and every detected face in a multi-person photo is checked independently. For an unfamiliar face, choose an existing identity, enter a new name, view the complete photo inside the app with a target reticle marking the current person, mark a false detection as **Not a face**, choose **I don't know this person** for a member of the general public, skip that face, or stop the scan with **Skip remaining / stop scan**. A normally skipped face is offered again on the next all-faces scan. An intentionally unknown person is placed in an anonymous biometric group, so matching appearances in later photos—and in future scans—do not prompt again. A later targeted reference can still identify that anonymous person. Names and anonymous groups learned early in a scan are used immediately on the remaining photos.

The identity dialog places the active face thumbnail beside an embedded full-image context view. It marks every detected face: red for unprocessed faces, cyan for selected targets, green with the saved name for identified people, and gray for intentionally unknown people. Click red reticles to select multiple targets, then mark the selected people as unknown, or use **I don't know anyone in this image** to classify every remaining unprocessed face separately. Detections marked **Not a face** are removed from later overlays.

While an unfamiliar-face question is open, background detection continues for up to 60 upcoming photos. The prompt shows a scrollable set of possible appearances of the active person and updates as more are detected. Saving an identity applies it to every included possible match. Click any thumbnail to toggle exclusion; excluded thumbnails display a red X and remain available for separate review. The queue is memory-bounded and freezes when the answer is submitted so unseen late arrivals are never assigned silently.

High-confidence automatic matches expand the active profile immediately, improving recognition of later angles and lighting conditions in the same scan. Automatic profile learning uses a stricter threshold than result searching, discards near-duplicate samples, and caps active profiles at 64 varied samples to reduce accidental profile drift.

Aligned face crops are checked for focus before profile learning. Blurry faces can still be assigned to named or anonymous identities and remain linked to their source photos, but their embeddings are excluded from active biometric profiles. The identity prompt labels these low-quality samples so the distinction is visible.

For drawings, paintings, statues, and other artwork, use **Save identity as art**. The face remains linked to the named identity and source image with a persistent artwork flag, but it is never admitted into the biometric profile—even when the artwork is sharp.

A lower threshold finds more possible matches but creates more false positives. A higher threshold is stricter but may miss the person. Different ages, angles, lighting, glasses, masks, and small or blurry faces affect accuracy.

## Privacy, consent, and limitations

Face embeddings are biometric data. Use this tool only on photos you are authorized to process and follow applicable consent, privacy, and retention requirements. Processing stays on the computer, but the first run downloads the MIT-licensed YuNet and SFace models from the official OpenCV Zoo.

The app stores preferences in `FacePhotoFinder/settings.json` and its biometric catalog in `FacePhotoFinder/face-catalog.sqlite3` inside the current user's local application-data folder. The catalog contains names, face embeddings, small aligned face previews, source paths and file fingerprints, and scan statistics. It does not copy full source photos. Treat the catalog as sensitive biometric data and protect or delete it according to your privacy and retention needs. Missing source files are ignored during searches.

Face recognition is probabilistic and can perform differently across demographic groups. Results are leads, not proof of identity. Review every result manually; do not use the tool for high-impact decisions, surveillance, or identification without consent.

## Tests

```powershell
python -m unittest discover -s tests -v
```
