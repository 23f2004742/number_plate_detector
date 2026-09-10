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

pip install -r requirements.txt
streamlit run app.py

Tesseract also needs to be installed on the operating system. On Hugging Face Spaces it is installed through packages.txt.
