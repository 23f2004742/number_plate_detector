---
title: Indian License Plate Recognition
emoji: 🚘
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.48.0"
app_file: app.py
pinned: false
---

Indian License Plate Recognition

A forensic-style vehicle license plate recovery pipeline, built around the same detection, restoration and OCR fusion approach validated in the training notebook, not just a single OCR call. It accepts a single photo, several photos of the same vehicle, or a short video, and combines everything it reads into one answer, marking any character it is not confident about instead of guessing.

Features

Trained YOLO plate detector with a fallback chain, retried at a lower confidence and larger input size, then a Haar cascade, then a contour-based guess, so a hard photo still returns a crop instead of nothing

Multiple photos or video frames of the same vehicle can be combined, since a single blurry frame is often not enough on its own

Quality-adaptive restoration, denoise, sharpen or exposure normalization is only applied when that specific defect is actually detected, plus perspective correction for angled plates

Two-line plate support for motorcycles and autos with a stacked plate layout

Tesseract and RapidOCR run over several image views, and all of their readings are fused character by character using sequence alignment, so agreement across readings counts for more than any single OCR pass

Common confusable characters, O and 0, I and 1, B and 8, S and 5, Z and 2, G and 6, are reconciled at the position level, and corrected against the Indian plate template when the position's expected type, letter or digit, is already known

Any vehicle's plate is read, not only the standard ten character Indian format, the standard format only earns a small confidence bonus, it is never required

Uploads in jpg, png, webp and avif all decode correctly

Pipeline

Upload a photo, a batch of photos, or a video of the vehicle.

Each frame goes through the detector fallback chain to find the plate.

The crop is checked for blur, exposure, noise, resolution and skew, then restored based on whichever of those is actually poor, and perspective corrected.

Tesseract and RapidOCR each read several views of the crop, plus the top and bottom halves separately when the plate looks like a two-line layout.

All of those readings are fused character by character, then, if more than one photo or frame was given, the per-photo results are fused again the same way.

The final confidence score combines OCR confidence, agreement across readings, character stability, image quality and geometric consistency. Below the confidence threshold the plate is reported as unreadable rather than guessed, and low-confidence characters are shown as a question mark rather than a false digit or letter.

Required files

app.py
requirements.txt
packages.txt
README.md
plate_detector.pt
pipeline_config.json

Notes

pipeline_config.json holds the same weights and thresholds produced by the training notebook, so the deployed app scores plates the same way the notebook evaluated them.

The system does not hard-code known plate numbers or blindly replace ambiguous characters. Confusable characters are only reconciled where the evidence and the plate template already support it.

Run locally

The instructions below assume Windows PowerShell. The same project also runs on macOS and Linux with equivalent Python, Git and Tesseract commands.

Prerequisites

- Git, available from https://git-scm.com/downloads
- Python 3.12 (64-bit), available from https://www.python.org/downloads/
- Tesseract OCR. On Windows, install it from https://github.com/UB-Mannheim/tesseract/wiki and add its installation directory to PATH. The default directory is usually `C:\Program Files\Tesseract-OCR`.

1. Clone the repository

On the repository page, select **Code**, copy the HTTPS URL, and run the following commands. Replace the URL with the repository's actual HTTPS URL if necessary.

```powershell
cd C:\src
git clone <REPOSITORY-HTTPS-URL> number-plate-detector
cd number-plate-detector
```

Keep the project path short on Windows. This helps avoid Windows path-length errors while PyTorch is installed.

2. Create and activate a virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once in PowerShell as the current user, then activate the environment again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

3. Install the Python dependencies

Run this as a separate command from the Streamlit command:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The installation can take several minutes because it includes PyTorch and computer-vision packages. If Windows reports `WinError 206` or `filename or extension is too long`, move or reclone the project to a shorter directory such as `C:\src\number-plate-detector`, then recreate `.venv` and repeat this step.

4. Check Tesseract

Close and reopen PowerShell after adding Tesseract to PATH, then run:

```powershell
tesseract --version
```

If the command is not recognized, add `C:\Program Files\Tesseract-OCR` to the Windows PATH and reopen PowerShell. The application can use RapidOCR as a fallback, but Tesseract is part of the intended OCR pipeline.

5. Start the application

```powershell
python -m streamlit run app.py
```

Open the URL printed in the terminal, normally http://localhost:8501. Keep the terminal open while using the application. Stop the server with `Ctrl+C`.

Do not combine the install and start commands. For example, `pip install -r requirements.txt streamlit run app.py` makes pip try to install `streamlit` and `app.py` as packages.

6. Use the application

Upload a JPG, PNG, WEBP or AVIF image, multiple images of the same vehicle, or a short video. The detector, restoration pipeline and OCR fusion will process the upload and report the recognized plate with confidence information.

Tesseract also needs to be installed on the operating system. On Hugging Face Spaces it is installed through `packages.txt`.
