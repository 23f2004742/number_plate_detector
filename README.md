License Plate Recognition

A vehicle license plate recognition pipeline using:

YOLO for license plate detection

OpenCV for crop processing, perspective correction and image enhancement

RapidOCR with PP-OCRv6 for primary OCR

Tesseract OCR as a fallback

Streamlit for the web interface

Pipeline

Upload a vehicle image.

YOLO detects the license plate.

The detected crop is processed through both the original and perspective-corrected paths.

Multiple faithful preprocessing variants are generated for difficult and blurred plates.

RapidOCR runs both text detection + recognition and recognition-only passes.

Tesseract provides independent fallback candidates.

Candidates are ranked using OCR confidence, repeated agreement and Indian registration-plate structure.

The detected plate and confidence metrics are displayed together with an annotated image.

Required files

Keep these files in the Space repository:

app.py
requirements.txt
apt.txt
plate_detector.pt
pipeline_config.json
README.md

plate_detector.pt and pipeline_config.json are required by the application and are not included in this README.

Notes

The OCR pipeline intentionally preserves grayscale information instead of relying only on aggressive thresholding. This is important for blurred plates because strong thresholding or sharpening can destroy character strokes that are still recoverable from the original crop.

Tesseract is installed through apt.txt and RapidOCR uses ONNX Runtime.