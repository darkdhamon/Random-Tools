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
- asks again when a new photo matches a previously anonymous person, with an option to update every earlier match;
- supports an hourly per-user background catalog task that stays hidden unless the same unidentified person appears in at least three distinct new photos;
- classifies explicit nudity locally with NudeNet and lets the web gallery hide, search, or manually correct NSFW results;
- explains NSFW decisions with each detected anatomical class, its confidence, and whether it contributed to the overall score;
- reduces shirtless-person false positives with deduplicated findings, class-specific thresholds, and male/female face context;
- uses the whole-image Falconsai ViT classifier as the automatic NSFW decision, retains NudeNet anatomical evidence, and routes unresolved model disagreements to the **NSFW conflicts** search until they are manually marked safe or NSFW;
- detects screenshots and document/receipt-style images locally, with independent hide filters, type searches, and manual corrections;
- presents the web gallery as a newest-first year/month/day timeline and automatically saves metadata edits after typing pauses;
- supports multi-select deletion with a danger confirmation and lets archived selections target an existing `.zip`/`.hide` archive or a newly named `.zip`, defaulting to `Hidden Pictures/GeneralArchive.zip`;
- applies **Mark safe** or **Mark NSFW** decisions to multiple selected gallery photos at once;
- includes Timeline and Identity web tabs with searchable identity statistics, biometric reference-image thumbnails and an in-app viewer, inline name/birth-year editing, person-filtered timeline navigation, removable false face detections, and searchable non-biometric manual person tags;
- extracts EXIF GPS coordinates, supports editable place names and coordinates, and shows an offline map preview with an explicit OpenStreetMap link;
- searches by a previously learned person from the **Known person** list without requiring another reference photo;
- autocompletes identity names case-insensitively while still allowing new names;
- shows up to three closest named-profile buttons with match percentages when automatic recognition is uncertain;
- provides a two-step danger-confirmed **Reset face database** action—including typing `RESET`—that removes all biometric and scan-cache data without touching source photos or saved preferences;
- includes **Manage identities** for reviewing face/photo assignments, moving selected entries, or merging an entire duplicate profile into an existing or newly named profile;
- estimates the apparent age of selected cataloged faces locally and stores the estimate for review;
- allows persistent capture-year overrides when EXIF is missing and a download timestamp gives the wrong year;
- fingerprints cataloged files so scans can relink moved photos without losing face assignments or metadata;
- marks deleted or otherwise unavailable source files as missing and provides a confirmed catalog-cleanup action;
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

On Windows, run `install-background-task.ps1` to install the hourly **Face Photo Finder Background Catalog** task. It scans the last folder selected in the app, relinks reorganized files, and stays hidden unless identification is needed. Task Scheduler ignores overlapping runs.

Run `install-gallery-service.ps1` to install the local web gallery at `http://127.0.0.1:8765`. It starts when you sign in and is reachable only from this computer. The gallery browses cataloged photos, filters by person or year, searches titles/tags/paths, edits titles, descriptions, tags, ratings and capture years, reassigns detected faces, and can start a catalog scan or open the full desktop interface.

## Use

1. To find a specific person, select clear reference photos or choose someone from **Known person**. If a reference contains multiple faces, choose one or several people from the thumbnail picker.
2. Select the folder containing the photo collection.
3. Leave match strictness at `0.45` initially and start the scan.
4. Review the ranked results. Double-click a row to open the original.
5. Export paths as CSV or copy matches to another folder. Copying preserves originals and resolves duplicate filenames automatically.

To build or update the identity catalog, clear the reference photos, leave **Known person** blank, and start a scan. Every photo containing a face is included, and every detected face in a multi-person photo is checked independently. For an unfamiliar face, choose an existing identity, enter a new name, view the complete photo inside the app with a target reticle marking the current person, mark a false detection as **Not a face**, choose **I don't know this person** for a member of the general public, skip that face, or stop the scan with **Skip remaining / stop scan**. A normally skipped face is offered again on the next all-faces scan. An intentionally unknown person is placed in an anonymous biometric group, so cached rescans do not repeat the same question. When a genuinely new or modified photo matches that group, the app asks once per scan whether you know the person now. You can name only the current appearance, or explicitly update every earlier anonymous match to the chosen profile. Names and anonymous groups learned early in a scan are used immediately on the remaining photos.

The identity dialog places the active face thumbnail beside an embedded full-image context view. It marks every detected face: red for unprocessed faces, cyan for selected targets, green with the saved name for identified people, and gray for intentionally unknown people. Click red reticles to select multiple targets, then mark the selected people as unknown, or use **I don't know anyone in this image** to classify every remaining unprocessed face separately. Detections marked **Not a face** are removed from later overlays.

While an unfamiliar-face question is open, background detection continues for up to 60 upcoming photos. The prompt shows a scrollable set of possible appearances of the active person and updates as more are detected. Saving an identity applies it to every included possible match. Click any thumbnail to toggle exclusion; excluded thumbnails display a red X and remain available for separate review. The queue is memory-bounded and freezes when the answer is submitted so unseen late arrivals are never assigned silently.

High-confidence automatic matches expand the active profile immediately, improving recognition of later angles and lighting conditions in the same scan. Automatic profile learning uses a stricter threshold than result searching, discards near-duplicate samples, and caps active profiles at 64 varied samples to reduce accidental profile drift.

Profiles are also organized by the photo's capture year. The app reads the EXIF date taken when available, then checks the filename for a four-digit year, and finally falls back to the file modification year. Matching uses samples from the same year first, or the nearest represented year when that year has no samples. The 64-sample profile is balanced across years so heavily photographed recent periods do not displace older appearances. In **Manage identities**, save a person's birth year to record the corresponding age for each photo year; birth year is descriptive metadata and does not weaken matching when it is unknown.

Downloaded files can have a misleading modification year. In **Manage identities**, select the affected face/photo rows and use **Set selected photo year** to save a permanent correction. **Estimate selected ages** downloads the Apache-licensed ONNX Model Zoo age classifier on first use, runs it locally, and shows the midpoint of a broad visual-age band. If the identity has a saved birth year, the app can suggest capture-year overrides, but it always asks before applying them. A `*` beside a year identifies a manual or confirmed override. Visual age is not exact and should be reviewed, especially for children and teenagers.

Each newly cataloged image receives a SHA-256 content fingerprint. At the beginning of a scan, the app compares missing catalog paths with files in the selected folder and relinks an unambiguous match in place, preserving identities, age estimates, artwork flags, and year overrides. Older catalog rows without a fingerprint use an exact filename-and-size match for their first relocation. Files that remain absent are marked **Missing** in **Manage identities**. **Remove missing entries** permanently removes those database records after confirmation without touching files. To discover a relocation, scan a folder that contains the file's new location.

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
