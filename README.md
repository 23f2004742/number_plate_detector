Indian License Plate Recognition

A robust vehicle license plate recognition pipeline designed to work with both clear and difficult vehicle images. The system uses a trained YOLO detector to locate the plate, OpenCV for crop correction and image enhancement, and RapidOCR with Tesseract as an independent OCR fallback.

Features

YOLO-based license plate detection

Perspective correction and safe crop padding

Multiple faithful image preprocessing paths

RapidOCR with PP-OCRv6 and ONNX Runtime

Tesseract fallback OCR

Position-aware Indian registration format validation

Independent recognition of plate groups (AA | 00 | AA | 0000)

Candidate consensus without hard-coding individual plate numbers

Designed for clear, low-resolution and blurred plates

Streamlit web interface

Pipeline

Upload a vehicle image.

YOLO detects the highest-confidence license plate.

The plate crop is preserved and also perspective-corrected.

OCR receives several faithful image variants, including grayscale, CLAHE, denoised, sharpened and illumination-normalized views.

RapidOCR and Tesseract generate independent readings.

The difficult AA00AA0000 plate structure is also evaluated group-by-group.

Candidates are ranked using OCR confidence, repeated agreement and valid Indian state-code/plate structure.

The detected plate, cleaned crop, annotated vehicle image and confidence metrics are displayed.

Required files

app.py
requirements.txt
apt.txt
README.md
plate_detector.pt
pipeline_config.json

Notes

The OCR stage deliberately keeps the original grayscale information. Strong thresholding and aggressive sharpening can remove character strokes from blurred plates, so they are not used as the sole recognition path.

The system does not hard-code known plate numbers or blindly replace ambiguous characters such as Q/O or B/U. Structured validation is used to rank genuine OCR evidence instead of inventing missing characters.

Run locally

pip install -r requirements.txt
streamlit run app.py

Tesseract must also be installed on the operating system. On Hugging Face Spaces it is installed through apt.txt.
