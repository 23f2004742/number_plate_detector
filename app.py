import os

import json
import html

# ============================================================
# Hugging Face ZeroGPU
# ============================================================

# AVIF support
import pillow_avif

# ============================================================
# Core libraries
# ============================================================

import cv2
import numpy as np
import pytesseract
import streamlit as st

# Modern OCR engine: PP-OCRv6 via RapidOCR. Tesseract remains as fallback.
try:
    from rapidocr import RapidOCR, LangRec, OCRVersion, ModelType
    RAPIDOCR_AVAILABLE = True
except Exception as e:
    RapidOCR = None
    LangRec = None
    OCRVersion = None
    ModelType = None
    RAPIDOCR_AVAILABLE = False
    print(f"RapidOCR unavailable, using Tesseract fallback: {e}")

from ultralytics import YOLO


# ============================================================
# Gradio 5.0.0 compatibility patch
# ============================================================

# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "plate_detector.pt"
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "pipeline_config.json"
)


# ============================================================
# Configuration
# ============================================================

print("Loading pipeline configuration...")

try:

    with open(CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)

except Exception as e:

    print(
        f"Warning: Could not load pipeline_config.json: {e}"
    )

    CONFIG = {
        "plate_size": [256, 96],

        "quality_weights": {
            "blur": 0.3,
            "exposure": 0.15,
            "noise": 0.15,
            "resolution": 0.2,
            "perspective": 0.1,
            "occlusion": 0.1
        },

        "score_weights": {
            "ocr": 0.3,
            "agreement": 0.2,
            "stability": 0.2,
            "quality": 0.15,
            "geometry": 0.15
        },

        "char_threshold": 0.35,
        "score_threshold": 0.25,
        "detector_trained": True
    }


PLATE_SIZE = tuple(
    CONFIG.get(
        "plate_size",
        [256, 96]
    )
)

QUALITY_WEIGHTS = CONFIG.get(
    "quality_weights",
    {}
)

SCORE_WEIGHTS = CONFIG.get(
    "score_weights",
    {}
)

CHAR_THRESHOLD = CONFIG.get(
    "char_threshold",
    0.35
)

SCORE_THRESHOLD = CONFIG.get(
    "score_threshold",
    0.25
)


# ============================================================
# Load trained detector
# ============================================================

@st.cache_resource(show_spinner=False)
def get_detector():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find trained detector: {MODEL_PATH}"
        )
    return YOLO(MODEL_PATH)

detector = get_detector()


# ============================================================
# Utilities
# ============================================================

def clamp(
    value,
    minimum=0.0,
    maximum=1.0
):

    return max(
        minimum,
        min(
            maximum,
            float(value)
        )
    )


def normalize_text(text):

    if text is None:
        return ""

    text = str(text).upper()

    allowed = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
    )

    return "".join(
        char
        for char in text
        if char in allowed
    )


# ============================================================
# Image quality
# ============================================================

def blur_score(image):

    if image is None or image.size == 0:
        return 0.0

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    variance = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    return clamp(
        variance / 500.0
    )


def exposure_score(image):

    if image is None or image.size == 0:
        return 0.0

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    mean_value = float(
        np.mean(gray)
    )

    distance = abs(
        mean_value - 127.5
    )

    return clamp(
        1.0 - distance / 127.5
    )


def noise_score(image):

    if image is None or image.size == 0:
        return 0.0

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    residual = cv2.absdiff(
        gray,
        blurred
    )

    noise = float(
        np.mean(residual)
    )

    return clamp(
        1.0 - noise / 50.0
    )


def resolution_score(image):

    if image is None or image.size == 0:
        return 0.0

    height, width = image.shape[:2]

    area = width * height

    return clamp(
        area / 15000.0
    )


def perspective_score(image):

    if image is None or image.size == 0:
        return 0.0

    height, width = image.shape[:2]

    if height == 0:
        return 0.0

    ratio = width / float(height)

    if ratio < 1.5:
        return 0.4

    if ratio > 8.0:
        return 0.5

    if 2.0 <= ratio <= 5.5:
        return 1.0

    if 1.5 <= ratio < 2.0:
        return 0.7

    return 0.75


def occlusion_score(image):

    if image is None or image.size == 0:
        return 0.0

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    std = float(
        np.std(gray)
    )

    return clamp(
        std / 80.0
    )


def calculate_quality_score(plate):

    scores = {
        "blur": blur_score(plate),
        "exposure": exposure_score(plate),
        "noise": noise_score(plate),
        "resolution": resolution_score(plate),
        "perspective": perspective_score(plate),
        "occlusion": occlusion_score(plate)
    }

    total = 0.0
    weight_sum = 0.0

    for key, score in scores.items():

        weight = QUALITY_WEIGHTS.get(
            key,
            0.0
        )

        total += score * weight
        weight_sum += weight

    if weight_sum == 0:
        return 0.0, scores

    return (
        clamp(
            total / weight_sum
        ),
        scores
    )


# ============================================================
# Plate processing
# ============================================================

def _order_quad_points(points):

    points = np.asarray(points, dtype=np.float32)

    s = points.sum(axis=1)
    d = np.diff(points, axis=1).reshape(-1)

    return np.array([
        points[np.argmin(s)],
        points[np.argmin(d)],
        points[np.argmax(s)],
        points[np.argmax(d)]
    ], dtype=np.float32)


def _find_plate_quad(plate):
    """Find the strongest quadrilateral corresponding to the plate face."""

    if plate is None or plate.size == 0:
        return None

    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) if len(plate.shape) == 3 else plate.copy()
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 140)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    image_area = float(h * w)

    best = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.18 or area > image_area * 0.98:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue

        for eps in (0.02, 0.03, 0.04, 0.05):
            approx = cv2.approxPolyDP(contour, eps * perimeter, True)
            if len(approx) != 4:
                continue

            pts = approx.reshape(4, 2).astype(np.float32)
            ordered = _order_quad_points(pts)

            width_top = np.linalg.norm(ordered[1] - ordered[0])
            width_bottom = np.linalg.norm(ordered[2] - ordered[3])
            height_left = np.linalg.norm(ordered[3] - ordered[0])
            height_right = np.linalg.norm(ordered[2] - ordered[1])

            width = (width_top + width_bottom) / 2.0
            height = (height_left + height_right) / 2.0
            if height <= 1 or width <= 1:
                continue

            aspect = width / height
            if aspect < 2.0 or aspect > 8.0:
                continue

            rectangularity = area / max(width * height, 1.0)
            score = (area / image_area) * 0.65 + rectangularity * 0.35

            if score > best_score:
                best_score = score
                best = ordered

    return best


def rectify_plate(plate):
    """Perspective-correct the YOLO crop, then resize for OCR."""

    if plate is None or plate.size == 0:
        return None

    target_width, target_height = PLATE_SIZE
    quad = _find_plate_quad(plate)

    if quad is not None:
        destination = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1]
        ], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(quad, destination)
        warped = cv2.warpPerspective(
            plate, matrix, (target_width, target_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
    else:
        padded = cv2.copyMakeBorder(
            plate, 8, 8, 12, 12,
            cv2.BORDER_REPLICATE
        )
        warped = cv2.resize(
            padded, (target_width, target_height),
            interpolation=cv2.INTER_CUBIC
        )

    return warped


# ============================================================
# OCR preprocessing
# ============================================================

def preprocess_plate_variants(plate):
    """
    Generate multiple preprocessing variants for OCR.

    The plate is enlarged before processing so Tesseract
    receives substantially more character information.
    """

    if plate is None or plate.size == 0:
        return []

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    if len(plate.shape) == 3:

        gray = cv2.cvtColor(
            plate,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = plate.copy()

    # --------------------------------------------------------
    # Upscale
    # --------------------------------------------------------

    gray = cv2.resize(
        gray,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )

    variants = []

    # --------------------------------------------------------
    # CLAHE
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    variants.append(
        (
            "clahe",
            enhanced
        )
    )

    # --------------------------------------------------------
    # Sharpen
    # --------------------------------------------------------

    sharpen_kernel = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ],
        dtype=np.float32
    )

    sharpened = cv2.filter2D(
        enhanced,
        -1,
        sharpen_kernel
    )

    variants.append(
        (
            "sharpened",
            sharpened
        )
    )

    # --------------------------------------------------------
    # OTSU
    # --------------------------------------------------------

    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants.append(
        (
            "otsu",
            otsu
        )
    )

    # --------------------------------------------------------
    # Adaptive threshold
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )

    variants.append(
        (
            "adaptive",
            adaptive
        )
    )

    # --------------------------------------------------------
    # Inverted OTSU
    # --------------------------------------------------------

    _, inverted = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    variants.append(
        (
            "inverted",
            inverted
        )
    )

    return variants


# ============================================================
# Indian plate format scoring
# ============================================================

def plate_format_score(text):
    """
    Score how closely an OCR candidate resembles
    a normal Indian vehicle registration format.
    """

    text = normalize_text(
        text
    )

    if not text:
        return 0.0

    score = 0.0

    # Typical lengths
    if len(text) == 10:

        score += 0.35

    elif len(text) == 9:

        score += 0.25

    elif len(text) == 8:

        score += 0.15

    # State code: first two letters
    if (
        len(text) >= 2
        and text[0].isalpha()
        and text[1].isalpha()
    ):

        score += 0.20

    # District code: next two digits
    if (
        len(text) >= 4
        and text[2].isdigit()
        and text[3].isdigit()
    ):

        score += 0.20

    # Last four characters commonly digits
    if (
        len(text) >= 8
        and all(
            c.isdigit()
            for c in text[-4:]
        )
    ):

        score += 0.25

    return clamp(
        score
    )


# ============================================================
# Levenshtein similarity
# ============================================================

def levenshtein_distance(a, b):

    if a == b:
        return 0

    if not a:
        return len(b)

    if not b:
        return len(a)

    previous = list(
        range(len(b) + 1)
    )

    for i, char_a in enumerate(
        a,
        start=1
    ):

        current = [i]

        for j, char_b in enumerate(
            b,
            start=1
        ):

            insert_cost = (
                current[j - 1] + 1
            )

            delete_cost = (
                previous[j] + 1
            )

            replace_cost = (
                previous[j - 1]
            )

            if char_a != char_b:
                replace_cost += 1

            current.append(
                min(
                    insert_cost,
                    delete_cost,
                    replace_cost
                )
            )

        previous = current

    return previous[-1]


def text_similarity(a, b):

    if not a or not b:
        return 0.0

    distance = levenshtein_distance(
        a,
        b
    )

    maximum = max(
        len(a),
        len(b)
    )

    if maximum == 0:
        return 1.0

    return clamp(
        1.0 - distance / maximum
    )


# ============================================================
# OCR
# ============================================================

INDIAN_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA",
    "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
    "TN", "TR", "TS", "UK", "UP", "WB"
}


def _clean_ocr_candidate(text):
    text = normalize_text(text)
    if len(text) < 5 or len(text) > 12:
        return ""
    return text


def _plate_structure_score(text):
    """Strict score for the Indian plate structure AA00AA0000."""
    text = normalize_text(text)
    if len(text) != 10:
        return 0.0

    score = 0.0
    if text[:2] in INDIAN_STATE_CODES:
        score += 0.45
    if text[2:4].isdigit():
        score += 0.20
    if text[4:6].isalpha():
        score += 0.15
    if text[6:10].isdigit():
        score += 0.20
    return score


def _ocr_variants(plate):
    """Create OCR inputs while preserving the original character information.

    Important: do not rely on a single thresholded image. On blurred plates,
    thresholding can destroy strokes that are still recoverable in grayscale.
    """
    if plate is None or plate.size == 0:
        return []

    if len(plate.shape) == 3:
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate.copy()

    h, w = gray.shape[:2]
    if h < 20 or w < 80:
        return []

    # A small border prevents characters touching the crop edge.
    gray = cv2.copyMakeBorder(gray, max(4, h // 18), max(4, h // 18),
                              max(6, w // 45), max(6, w // 45),
                              cv2.BORDER_REPLICATE)

    # Keep enough pixels for OCR. 4x is deliberately used for small plates.
    gray = cv2.resize(gray, None, fx=4.0, fy=4.0,
                      interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(gray)
    bilateral = cv2.bilateralFilter(clahe, 5, 30, 30)

    # Mild unsharp mask. Strong sharpening creates fake character strokes.
    soft = cv2.GaussianBlur(bilateral, (0, 0), 1.0)
    sharp = cv2.addWeighted(bilateral, 1.35, soft, -0.35, 0)

    # Illumination normalization.
    bg = cv2.GaussianBlur(clahe, (0, 0), 15)
    normalized = cv2.divide(clahe, bg, scale=180)

    _, otsu = cv2.threshold(normalized, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 41, 9
    )

    variants = [
        ("original", gray),
        ("clahe", clahe),
        ("bilateral", bilateral),
        ("sharp", sharp),
        ("normalized", normalized),
        ("otsu", otsu),
        ("adaptive", adaptive),
    ]

    return variants


@st.cache_resource(show_spinner=False)
def _get_rapidocr():
    if not RAPIDOCR_AVAILABLE:
        return None
    try:
        # rapidocr >= 3.9 defaults to PP-OCRv6 small + ONNX Runtime.
        return RapidOCR()
    except Exception as e:
        print(f"RapidOCR initialization failed: {e}")
        return None


def _rapidocr_read(image):
    """Run RapidOCR in its normal detection+recognition pipeline first.

    Recognition-only is useful when the crop is perfect, but the full OCR
    pipeline is more robust when blur causes the text-line boundaries to be
    uncertain. We therefore try both and keep the returned candidates.
    """
    engine = _get_rapidocr()
    if engine is None:
        return []

    candidates = []

    # Normal RapidOCR pipeline: detector + recognizer.
    try:
        result = engine(image)
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        if texts is not None:
            for i, text in enumerate(texts):
                clean = _clean_ocr_candidate(text)
                if not clean:
                    continue
                score = float(scores[i]) if scores is not None and i < len(scores) else 0.0
                candidates.append((clean, score, "det-rec"))
    except Exception as e:
        print(f"RapidOCR full pipeline failed: {e}")

    # Recognition-only pass over the whole plate line.
    try:
        result = engine(image, use_det=False, use_cls=False, use_rec=True)
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        if texts is not None:
            for i, text in enumerate(texts):
                clean = _clean_ocr_candidate(text)
                if not clean:
                    continue
                score = float(scores[i]) if scores is not None and i < len(scores) else 0.0
                candidates.append((clean, score, "rec-only"))
    except Exception as e:
        print(f"RapidOCR recognition-only failed: {e}")

    return candidates


def _tesseract_read(image):
    results = []
    for psm in (6, 7, 8, 11, 13):
        config = (
            f"--psm {psm} "
            "-l eng "
            "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        try:
            text = _clean_ocr_candidate(
                pytesseract.image_to_string(image, config=config)
            )
            if not text:
                continue

            confidence_values = []
            try:
                data = pytesseract.image_to_data(
                    image, config=config,
                    output_type=pytesseract.Output.DICT
                )
                for value in data.get("conf", []):
                    try:
                        value = float(value)
                        if value >= 0:
                            confidence_values.append(value)
                    except Exception:
                        pass
            except Exception:
                pass

            confidence = (
                float(np.mean(confidence_values)) / 100.0
                if confidence_values else 0.30
            )
            results.append((text, confidence, psm))
        except Exception:
            pass

    return results


def _normalize_by_plate_position(text):
    """Repair only impossible type errors; never Q/O/G/B substitutions."""
    text = normalize_text(text)
    if len(text) != 10:
        return text

    chars = list(text)

    to_digit = {"O": "0", "D": "0", "I": "1", "L": "1",
                "Z": "2", "S": "5", "G": "6", "T": "7", "B": "8"}
    for i in (2, 3, 6, 7, 8, 9):
        if chars[i].isalpha() and chars[i] in to_digit:
            chars[i] = to_digit[chars[i]]

    to_letter = {"0": "O", "1": "I", "2": "Z", "5": "S",
                 "6": "G", "7": "T", "8": "B"}
    for i in (0, 1, 4, 5):
        if chars[i].isdigit() and chars[i] in to_letter:
            chars[i] = to_letter[chars[i]]

    return "".join(chars)


def _score_candidate(text, source_conf=0.0):
    text = _normalize_by_plate_position(text)
    if len(text) != 10:
        # Partial readings are retained for diagnostics but should not beat a
        # complete plate reading.
        return 0.12 * clamp(source_conf)

    state = text[:2] in INDIAN_STATE_CODES
    district = text[2:4].isdigit()
    series = text[4:6].isalpha()
    number = text[6:10].isdigit()

    structure = (0.40 * state + 0.20 * district +
                 0.15 * series + 0.25 * number)
    return 0.62 * structure + 0.38 * clamp(source_conf)


def _prepare_ocr_sources(plate, rectified=None):
    """Use both the raw detector crop and rectified crop.

    A contour-based perspective correction can be wrong on blurred plates.
    Keeping the raw crop gives OCR a second chance using the pixels YOLO
    actually detected.
    """
    sources = []
    seen_shapes = set()

    for name, image in (("raw", plate), ("rectified", rectified)):
        if image is None or image.size == 0:
            continue
        key = (name, image.shape[:2])
        if key in seen_shapes:
            continue
        seen_shapes.add(key)
        for variant_name, variant in _ocr_variants(image):
            sources.append((f"{name}-{variant_name}", variant))

    return sources


def run_ocr(plate, rectified=None):
    """Robust OCR using multiple faithful views of the detected plate."""
    if plate is None or plate.size == 0:
        return [], None

    sources = _prepare_ocr_sources(plate, rectified)
    results = []

    for variant_name, image in sources:
        for text, confidence, engine_mode in _rapidocr_read(image):
            results.append({
                "text": _normalize_by_plate_position(text),
                "confidence": confidence * 100.0,
                "variant": f"rapidocr-{variant_name}-{engine_mode}",
                "psm": 0,
                "image": image,
                "engine": "rapidocr",
            })

        for text, confidence, psm in _tesseract_read(image):
            results.append({
                "text": _normalize_by_plate_position(text),
                "confidence": confidence * 100.0,
                "variant": f"tesseract-{variant_name}",
                "psm": psm,
                "image": image,
                "engine": "tesseract",
            })

    if not results:
        return [], None

    # Group exact normalized readings. Do NOT merge different characters or
    # synthesize a plate from unrelated candidates.
    grouped = {}
    for r in results:
        text = normalize_text(r.get("text", ""))
        if not text:
            continue
        item = grouped.setdefault(text, {
            "count": 0,
            "conf": [],
            "rapid": 0,
            "tess": 0,
            "examples": []
        })
        item["count"] += 1
        item["conf"].append(clamp(r.get("confidence", 0.0) / 100.0))
        if r.get("engine") == "rapidocr":
            item["rapid"] += 1
        else:
            item["tess"] += 1
        item["examples"].append(r)

    best = None
    best_score = -1.0

    for text, info in grouped.items():
        mean_conf = float(np.mean(info["conf"])) if info["conf"] else 0.0
        complete = len(text) == 10
        structure = _score_candidate(text, mean_conf)

        # Independent engine agreement is useful. It is only a bonus and can
        # never override an invalid plate structure.
        engine_agreement = 1.0 if info["rapid"] > 0 and info["tess"] > 0 else 0.0
        repetition = min(info["count"] / 5.0, 1.0)

        score = (
            0.62 * structure
            + 0.16 * repetition
            + 0.12 * engine_agreement
            + 0.10 * mean_conf
        )

        if complete and text[:2] in INDIAN_STATE_CODES:
            score += 0.12
        elif complete:
            score -= 0.12

        # Never let an incomplete candidate win over a valid complete plate.
        if not complete:
            score -= 0.25

        if score > best_score:
            best_score = score
            example = max(info["examples"], key=lambda r: r["confidence"])
            best = {
                **example,
                "text": text,
                "combined_score": clamp(score),
                "agreement": repetition,
            }

    return results, best

def select_best_ocr_result(results):
    """Compatibility wrapper for older callers."""
    if not results:
        return None
    best = None
    best_score = -1.0
    for result in results:
        text = _normalize_by_plate_position(result.get("text", ""))
        score = _score_candidate(text, result.get("confidence", 0.0) / 100.0)
        if score > best_score:
            best_score = score
            best = {**result, "text": text, "combined_score": clamp(score)}
    return best


# ============================================================
# YOLO detection
# ============================================================

def detect_plate(image):

    if image is None or image.size == 0:

        return (
            None,
            None,
            0.0
        )

    try:

        results = detector.predict(
            source=image,
            verbose=False
        )

    except Exception as e:

        print(
            f"Detection error: {e}"
        )

        return (
            None,
            None,
            0.0
        )

    if not results:

        return (
            None,
            None,
            0.0
        )

    result = results[0]

    if result.boxes is None:

        return (
            None,
            None,
            0.0
        )

    if len(result.boxes) == 0:

        return (
            None,
            None,
            0.0
        )

    best_index = 0
    best_confidence = 0.0

    for i, box in enumerate(
        result.boxes
    ):

        try:

            confidence = float(
                box.conf[0].item()
            )

        except Exception:

            confidence = 0.0

        if confidence > best_confidence:

            best_confidence = confidence
            best_index = i

    box = result.boxes[
        best_index
    ]

    xyxy = (
        box.xyxy[0]
        .cpu()
        .numpy()
    )

    x1, y1, x2, y2 = map(
        int,
        xyxy
    )

    height, width = image.shape[:2]

    x1 = max(
        0,
        min(
            x1,
            width - 1
        )
    )

    y1 = max(
        0,
        min(
            y1,
            height - 1
        )
    )

    x2 = max(
        0,
        min(
            x2,
            width
        )
    )

    y2 = max(
        0,
        min(
            y2,
            height
        )
    )

    if x2 <= x1 or y2 <= y1:

        return (
            None,
            None,
            best_confidence
        )

    plate = image[
        y1:y2,
        x1:x2
    ].copy()

    return (
        plate,
        (
            x1,
            y1,
            x2,
            y2
        ),
        best_confidence
    )




# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="License Plate Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background: #080d16; color: #e8edf5; }
    .block-container { max-width: 1240px; padding-top: 2rem; }
    .title { font-size: 2.1rem; font-weight: 700; margin-bottom: .2rem; }
    .subtitle { color: #8e9bb0; margin-bottom: 1.2rem; }
    .result-box { border: 1px solid #26364d; border-radius: 12px; background: #0d1624; padding: 20px; text-align: center; }
    .plate { color: #35e58d; font: 700 2.2rem 'Courier New', monospace; letter-spacing: 3px; }
    .metric { border: 1px solid #26364d; border-radius: 10px; background: #0a111c; padding: 15px; }
    .metric-label { color: #8190a7; font-size: .78rem; }
    .metric-value { font-size: 1.35rem; font-weight: 700; margin-top: 4px; }
    .candidate { color: #aab6c8; font-family: monospace; font-size: .85rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">License Plate Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect and read vehicle license plates using YOLO, OpenCV and OCR.</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png", "webp", "avif"],
    help="For best results, upload an image where the vehicle plate is visible. The pipeline also tries multiple views for blurred plates.",
)

if uploaded is None:
    st.info("Upload a vehicle image to start recognition.")
else:
    try:
        file_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image_bgr is None:
            st.error("Could not decode the uploaded image.")
            st.stop()
    except Exception as e:
        st.error(f"Could not read image: {e}")
        st.stop()

    run = st.button("Recognize Plate", type="primary", use_container_width=True)

    if run:
        with st.spinner("Detecting plate and running OCR..."):
            original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            plate, bbox, detector_confidence = detect_plate(image_bgr)

            if plate is None:
                st.image(original_rgb, caption="Uploaded Image", use_container_width=True)
                st.error("No license plate detected.")
                st.stop()

            quality, _ = calculate_quality_score(plate)
            rectified = rectify_plate(plate)
            ocr_results, best_ocr = run_ocr(plate, rectified)

            if best_ocr is not None:
                text = normalize_text(best_ocr["text"])
                agreement = float(best_ocr.get("agreement", 0.0))
                ocr_confidence = clamp(float(best_ocr.get("confidence", 0.0)) / 100.0)
                cleaned = best_ocr["image"]
            else:
                text = ""
                agreement = 0.0
                ocr_confidence = 0.0
                variants = preprocess_plate_variants(rectified)
                cleaned = variants[0][1] if variants else rectified

            geometry = perspective_score(plate)
            final_score = (
                SCORE_WEIGHTS.get("ocr", 0.3) * ocr_confidence
                + SCORE_WEIGHTS.get("agreement", 0.2) * agreement
                + SCORE_WEIGHTS.get("stability", 0.2) * agreement
                + SCORE_WEIGHTS.get("quality", 0.15) * quality
                + SCORE_WEIGHTS.get("geometry", 0.15) * geometry
            )
            final_score = clamp(final_score)

            annotated = image_bgr.copy()
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
            label = f"{text}  {detector_confidence:.0%}" if text else "PLATE DETECTED"
            (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            label_y = max(30, y1)
            cv2.rectangle(annotated, (x1, label_y - lh - 15), (x1 + lw + 12, label_y), (0, 180, 0), -1)
            cv2.putText(annotated, label, (x1 + 6, label_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(original_rgb, caption="Uploaded Image", use_container_width=True)
        with c2:
            if cleaned is not None:
                if len(cleaned.shape) == 2:
                    cleaned_display = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
                else:
                    cleaned_display = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
                st.image(cleaned_display, caption="Detected Plate (Cleaned)", use_container_width=True)
        with c3:
            st.image(annotated_rgb, caption="Annotated Image", use_container_width=True)

        if text:
            st.markdown(
                f'<div class="result-box"><div style="color:#8090a8;font-size:.75rem;letter-spacing:1px">DETECTED PLATE NUMBER</div><div class="plate">{text}</div></div>',
                unsafe_allow_html=True,
            )
            st.write("")
            m1, m2, m3, m4 = st.columns(4)
            for col, label, value in [
                (m1, "Overall Confidence", final_score),
                (m2, "Detection Confidence", detector_confidence),
                (m3, "OCR Confidence", ocr_confidence),
                (m4, "Image Quality", quality),
            ]:
                with col:
                    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value:.1%}</div></div>', unsafe_allow_html=True)

            candidates = list(dict.fromkeys([normalize_text(x.get("text", "")) for x in ocr_results if x.get("text")]))[:10]
            if candidates:
                st.markdown("**OCR candidates:** " + "  ".join([f'<span class="candidate">{x}</span>' for x in candidates]), unsafe_allow_html=True)
        else:
            st.warning("Plate detected, but OCR could not produce a reliable reading.")
