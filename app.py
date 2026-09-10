import os
import json
import base64
import tempfile

import cv2
import numpy as np
import pytesseract
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

try:
    import pillow_avif  # noqa: F401, adds AVIF support to Pillow
except Exception:
    pass

try:
    from rapidocr import RapidOCR
    RAPIDOCR_AVAILABLE = True
except Exception as e:
    RapidOCR = None
    RAPIDOCR_AVAILABLE = False
    print(f"RapidOCR unavailable, tesseract only: {e}")

from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plate_detector.pt")
CONFIG_PATH = os.path.join(BASE_DIR, "pipeline_config.json")

DEFAULT_CONFIG = {
    "plate_size": [256, 96],
    "quality_weights": {"blur": 0.30, "exposure": 0.15, "noise": 0.15,
                         "resolution": 0.20, "perspective": 0.10, "occlusion": 0.10},
    "score_weights": {"ocr": 0.30, "agreement": 0.20, "stability": 0.20,
                       "quality": 0.15, "geometry": 0.15},
    "char_threshold": 0.35,
    "score_threshold": 0.25,
}

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = {**DEFAULT_CONFIG, **json.load(f)}
except Exception:
    CONFIG = DEFAULT_CONFIG

PLATE_SIZE = tuple(CONFIG.get("plate_size", [256, 96]))
QUALITY_WEIGHTS = CONFIG.get("quality_weights", DEFAULT_CONFIG["quality_weights"])
SCORE_WEIGHTS = CONFIG.get("score_weights", DEFAULT_CONFIG["score_weights"])
CHAR_THRESHOLD = CONFIG.get("char_threshold", 0.35)
SCORE_THRESHOLD = CONFIG.get("score_threshold", 0.25)

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
SIMILAR = {"O": ["0"], "0": ["O"], "I": ["1"], "1": ["I"], "B": ["8"], "8": ["B"],
           "S": ["5"], "5": ["S"], "Z": ["2"], "2": ["Z"], "G": ["6"], "6": ["G"]}
INDIAN_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HR", "HP",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD",
    "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(v)))


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).upper()
    return "".join(c for c in text if c in CHARS)


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_detector():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Could not find trained detector: {MODEL_PATH}")
    return YOLO(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_haar():
    path = os.path.join(getattr(cv2.data, "haarcascades", ""), "haarcascade_russian_plate_number.xml")
    if os.path.isfile(path) and hasattr(cv2, "CascadeClassifier"):
        c = cv2.CascadeClassifier(path)
        if not c.empty():
            return c
    return None


@st.cache_resource(show_spinner=False)
def get_rapidocr():
    if not RAPIDOCR_AVAILABLE:
        return None
    try:
        return RapidOCR()
    except Exception as e:
        print(f"RapidOCR init failed: {e}")
        return None


detector = get_detector()
haar = get_haar()


# ---------------------------------------------------------------------------
# detection, with a fallback chain so a real vehicle photo almost never
# comes back with nothing to work on, even when the trained detector
# misses it (small, blurry or unusually angled plate)
# ---------------------------------------------------------------------------

def _contour_guess(image):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.dilate(cv2.Canny(gray, 50, 150), np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, -1
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < 25 or ch < 10 or not (1.3 <= cw / ch <= 8.0):
            continue
        if cw * ch > best_area:
            best_area, best = cw * ch, (x, y, x + cw, y + ch)
    return best


def detect_plate(image):
    """Returns (crop, bbox, confidence, backend). Tries the trained YOLO
    detector first (twice, the second time at a lower confidence and a
    larger inference size, which is what actually finds a small or blurred
    plate). Falls back to a classical Haar cascade, then a contour guess,
    so a photo that clearly contains a vehicle is very rarely rejected
    outright."""
    h, w = image.shape[:2]

    for conf, imgsz in ((0.25, 640), (0.10, 1280)):
        try:
            results = detector.predict(source=image, conf=conf, imgsz=imgsz, verbose=False)
        except Exception as e:
            print(f"detector error: {e}")
            results = None
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confs = [float(b.conf[0].item()) for b in boxes]
            i = int(np.argmax(confs))
            x1, y1, x2, y2 = map(int, boxes[i].xyxy[0].cpu().numpy())
            x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
            x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
            if x2 > x1 and y2 > y1:
                return image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2), confs[i], "yolo"

    if haar is not None:
        boxes = haar.detectMultiScale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 1.1, 4)
        if len(boxes) > 0:
            x, y, cw, ch = max(boxes, key=lambda b: b[2] * b[3])
            x1, y1, x2, y2 = x, y, x + cw, y + ch
            return image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2), 0.35, "haar"

    guess = _contour_guess(image)
    if guess is not None:
        x1, y1, x2, y2 = guess
        return image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2), 0.20, "contour"

    return None, None, 0.0, "none"


# ---------------------------------------------------------------------------
# quality, rectification and cleanup
# ---------------------------------------------------------------------------

def check_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]

    blur = clamp(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0)
    exposure = clamp(1.0 - abs(float(np.mean(gray)) - 127.5) / 127.5)
    residual = cv2.absdiff(gray, cv2.GaussianBlur(gray, (3, 3), 0))
    noise = clamp(1.0 - float(np.mean(residual)) / 50.0)
    resolution = clamp((w * h) / 15000.0)
    ratio = w / float(h) if h else 0.0
    if 2.0 <= ratio <= 5.5:
        perspective = 1.0
    elif 1.2 <= ratio < 2.0:
        perspective = 0.7
    elif ratio > 8.0:
        perspective = 0.5
    else:
        perspective = 0.4
    occlusion = clamp(float(np.std(gray)) / 80.0)

    scores = {"blur": blur, "exposure": exposure, "noise": noise,
              "resolution": resolution, "perspective": perspective, "occlusion": occlusion}
    total = sum(scores[k] * QUALITY_WEIGHTS.get(k, 0.0) for k in scores)
    weight_sum = sum(QUALITY_WEIGHTS.get(k, 0.0) for k in scores)
    scores["combined"] = clamp(total / weight_sum) if weight_sum else 0.0
    return scores


def _order_quad(points):
    points = np.asarray(points, dtype=np.float32)
    s = points.sum(axis=1)
    d = np.diff(points, axis=1).reshape(-1)
    return np.array([points[np.argmin(s)], points[np.argmin(d)],
                      points[np.argmax(s)], points[np.argmax(d)]], dtype=np.float32)


def _find_plate_quad(plate):
    """Look for the plate's own rectangle inside the crop, so a skewed
    photo still gets straightened instead of just resized."""
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) if plate.ndim == 3 else plate.copy()
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    area_total = float(h * w)
    best, best_score = None, -1.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < area_total * 0.18 or area > area_total * 0.98:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        for eps in (0.02, 0.03, 0.04):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) != 4:
                continue
            q = _order_quad(approx.reshape(4, 2))
            width = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
            height = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
            if height <= 1 or width <= 1:
                continue
            aspect = width / height
            if not (1.2 <= aspect <= 8.0):
                continue
            rectangularity = area / max(width * height, 1.0)
            score = (area / area_total) * 0.65 + rectangularity * 0.35
            if score > best_score:
                best_score, best = score, q
    return best


def rectify(plate):
    if plate is None or plate.size == 0:
        return None
    tw, th = PLATE_SIZE
    quad = _find_plate_quad(plate)
    if quad is not None:
        dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(quad, dst)
        return cv2.warpPerspective(plate, matrix, (tw, th), flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
    padded = cv2.copyMakeBorder(plate, 8, 8, 12, 12, cv2.BORDER_REPLICATE)
    return cv2.resize(padded, (tw, th), interpolation=cv2.INTER_CUBIC)


def ocr_variants(image, quality):
    """A handful of different views of the same crop. Different
    degradations respond to different processing, a blurred plate is
    often more readable in plain grayscale than after aggressive
    thresholding, so every view is tried and OCR is left to see what it
    can read from each of them."""
    if image is None or image.size == 0:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    h, w = gray.shape[:2]
    gray = cv2.copyMakeBorder(gray, max(4, h // 20), max(4, h // 20),
                               max(6, w // 45), max(6, w // 45), cv2.BORDER_REPLICATE)
    scale = 6.0 if quality["resolution"] < 0.4 else 4.0
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants = [("gray", gray), ("otsu", otsu)]

    # only one extra view is added, matched to whichever defect is worst,
    # instead of stacking every enhancement on every crop, since that was
    # multiplying OCR calls far past what a live app can serve quickly
    worst_name, worst_score = min(
        (("noise", quality["noise"]), ("blur", quality["blur"]), ("exposure", quality["exposure"])),
        key=lambda kv: kv[1],
    )
    if worst_score < 0.6:
        if worst_name == "noise":
            variants.append(("denoise", cv2.bilateralFilter(clahe, 5, 30, 30)))
        elif worst_name == "blur":
            soft = cv2.GaussianBlur(clahe, (0, 0), 1.2)
            variants.append(("sharp", cv2.addWeighted(clahe, 1.6, soft, -0.6, 0)))
        else:
            background = cv2.GaussianBlur(clahe, (0, 0), 15)
            variants.append(("normalized", cv2.divide(clahe, background, scale=180)))
    else:
        variants.append(("clahe", clahe))

    return variants


def split_two_line(image):
    """Many two-wheelers carry a stacked, two-line plate. A single-line
    OCR pass reads that as one garbled line, so the top and bottom halves
    are also tried separately when the crop is squarish rather than the
    usual wide single-line shape."""
    h, w = image.shape[:2]
    if h == 0 or w / float(h) > 2.3:
        return None
    mid = h // 2
    pad = max(2, h // 12)
    top = image[: mid + pad, :]
    bottom = image[max(0, mid - pad):, :]
    return top, bottom


# ---------------------------------------------------------------------------
# OCR engines
# ---------------------------------------------------------------------------

def _tesseract_read(image):
    # image_to_data alone gives text and per-word confidence in one pass,
    # a second image_to_string call on the same config was pure waste
    out = []
    for psm in (7, 8, 6):
        config = f"--psm {psm} -c tessedit_char_whitelist={CHARS}"
        try:
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        except Exception:
            continue
        text = normalize_text("".join(data.get("text", [])))
        if not text:
            continue
        values = [float(c) for c in data.get("conf", []) if c not in ("-1", -1)]
        conf = (float(np.mean(values)) / 100.0) if values else 0.3
        out.append((text, max(conf, 0.15)))
    return out


def _rapidocr_read(image):
    # the crop given here is already localized to the plate (and, for the
    # two-line case, to one line of it), so a plain recognition pass is
    # used instead of running full detection on top of an already-cropped
    # region, which only doubled the work for no real gain
    engine = get_rapidocr()
    if engine is None:
        return []
    out = []
    try:
        result = engine(image, use_det=False, use_cls=False, use_rec=True)
    except Exception as e:
        print(f"rapidocr failed: {e}")
        return out
    texts = getattr(result, "txts", None) or []
    scores = getattr(result, "scores", None) or []
    for i, text in enumerate(texts):
        text = normalize_text(text)
        if text:
            conf = float(scores[i]) if i < len(scores) else 0.5
            out.append((text, max(conf, 0.15)))
    return out


# ---------------------------------------------------------------------------
# character level fusion, the same algorithm used to combine multiple
# photos of one plate is reused here to combine multiple OCR readings of
# one photo, and again to combine multiple photos of the same vehicle
# ---------------------------------------------------------------------------

def _align(ref, seq, gap=-1, match=2, mismatch=-1):
    n, m = len(ref), len(seq)
    dp = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        dp[i][0] = i * gap
    for j in range(m + 1):
        dp[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if ref[i - 1] == seq[j - 1] else mismatch
            dp[i][j] = max(dp[i-1][j-1] + s, dp[i-1][j] + gap, dp[i][j-1] + gap)
    i, j, pairs = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = match if ref[i-1] == seq[j-1] else mismatch
            if dp[i][j] == dp[i-1][j-1] + s:
                pairs.append((i - 1, seq[j - 1])); i -= 1; j -= 1; continue
        if i > 0 and dp[i][j] == dp[i-1][j] + gap:
            i -= 1; continue
        j -= 1
    return list(reversed(pairs))


def combine_readings(readings, credit=0.4):
    # readings: list of dict with text, conf, weight
    readings = [r for r in readings if r["text"]]
    if not readings:
        return "", []
    # the reference sequence anchors every alignment below, so it is
    # picked by which length carries the most weighted evidence, not by
    # a plain median of lengths, since a couple of noisy reads that
    # happen to share a length can otherwise outvote the real length
    length_weight = {}
    for r in readings:
        L = len(r["text"])
        length_weight[L] = length_weight.get(L, 0.0) + r["weight"] * r["conf"]
    ref_len = max(length_weight, key=length_weight.get)
    same_len = [r for r in readings if len(r["text"]) == ref_len]
    ref = max(same_len, key=lambda r: r["weight"] * r["conf"])["text"]

    votes = [dict() for _ in range(len(ref))]
    for r in readings:
        for pos, ch in _align(ref, r["text"]):
            if pos < len(votes):
                votes[pos][ch] = votes[pos].get(ch, 0.0) + r["weight"] * r["conf"]

    for pos_votes in votes:
        if len(pos_votes) < 2:
            continue
        snapshot = dict(pos_votes)
        extra = {}
        for ch, w in snapshot.items():
            for alt in SIMILAR.get(ch, []):
                if alt in snapshot and alt != ch:
                    extra[alt] = extra.get(alt, 0.0) + w * credit
        for alt, w in extra.items():
            pos_votes[alt] = pos_votes.get(alt, 0.0) + w

    out_text, out_conf = [], []
    for pos_votes in votes:
        if not pos_votes:
            out_text.append("?"); out_conf.append(0.0); continue
        total = sum(pos_votes.values())
        best_ch, best_w = max(pos_votes.items(), key=lambda kv: kv[1])
        c = best_w / total if total else 0.0
        out_text.append(best_ch if c >= CHAR_THRESHOLD else "?")
        out_conf.append(c)
    return "".join(out_text), out_conf


def reweight(readings, fused_text):
    out = []
    for r in readings:
        n = min(len(r["text"]), len(fused_text))
        agree = (sum(a == b for a, b in zip(r["text"][:n], fused_text[:n])) / n) if n else 0.0
        out.append({**r, "weight": r["weight"] * max(0.2, agree)})
    return out


def fuse(readings):
    first, _ = combine_readings(readings)
    if not first:
        return "", []
    return combine_readings(reweight(readings, first))


def repair_format(text):
    """Indian plates follow a fixed letter/digit template (state code,
    RTO code, series, number). Most OCR misses at this point are a single
    confusable character sitting in a position whose class is already
    known from the template, so this swaps it for the confusable of the
    right class instead of leaving an otherwise-correct read wrong. It
    only ever swaps within the existing confusable pairs, never guesses
    a fresh character, and does nothing to text that does not fit the
    template lengths."""
    if len(text) == 10:
        template = "LLDDLLDDDD"
    elif len(text) == 9:
        template = "LLDDLDDDD"
    else:
        return text
    out = list(text)
    for i, want in enumerate(template):
        ch = out[i]
        is_digit = ch.isdigit()
        if want == "D" and not is_digit:
            for alt in SIMILAR.get(ch, []):
                if alt.isdigit():
                    out[i] = alt
                    break
        elif want == "L" and is_digit:
            for alt in SIMILAR.get(ch, []):
                if alt.isalpha():
                    out[i] = alt
                    break
    return "".join(out)


def indian_format_bonus(text):
    """A small, optional nudge, never a hard requirement. Plates that are
    not this exact Indian format (bikes with two-line plates, BH-series,
    older formats, other countries) still get a normal reading."""
    if len(text) != 10:
        return 0.0
    bonus = 0.0
    if text[:2] in INDIAN_STATE_CODES:
        bonus += 0.06
    if text[2:4].isdigit():
        bonus += 0.02
    if text[6:10].isdigit():
        bonus += 0.02
    return bonus


# ---------------------------------------------------------------------------
# per image and per group pipeline
# ---------------------------------------------------------------------------

def read_plate(plate, rectified, quality):
    """Runs every OCR engine over every preprocessing view of one crop
    (plus a two-line split for squarish plates) and fuses all of it into
    a single reading for this one photo."""
    sources = []
    for name, img in (("raw", plate), ("rectified", rectified)):
        if img is None or img.size == 0:
            continue
        for vname, variant in ocr_variants(img, quality):
            sources.append((f"{name}-{vname}", variant))

    two_line = split_two_line(rectified) if rectified is not None else None
    if two_line is not None:
        top, bottom = two_line
        for half_name, half in (("top", top), ("bottom", bottom)):
            for vname, variant in ocr_variants(half, quality):
                sources.append((f"line-{half_name}-{vname}", variant))

    readings = []
    for name, img in sources:
        for text, conf in _tesseract_read(img):
            readings.append({"text": text, "conf": conf, "weight": 1.0, "source": f"tess-{name}"})
        for text, conf in _rapidocr_read(img):
            readings.append({"text": text, "conf": conf, "weight": 1.0, "source": f"rapid-{name}"})

    if not readings:
        return "", [], []

    # reading two half-plate texts back to back is a plausible extra
    # candidate for a stacked plate, added alongside the normal readings
    if two_line is not None:
        top_texts = [r["text"] for r in readings if "line-top" in r["source"]]
        bottom_texts = [r["text"] for r in readings if "line-bottom" in r["source"]]
        if top_texts and bottom_texts:
            combo = max(top_texts, key=len) + max(bottom_texts, key=len)
            readings.append({"text": combo, "conf": 0.4, "weight": 1.0, "source": "two-line-combo"})

    text, char_conf = fuse(readings)
    return text, char_conf, readings


def confidence_score(ocr_conf, agreement, stability, quality, geometry):
    weights = SCORE_WEIGHTS
    parts = {"ocr": ocr_conf, "agreement": agreement, "stability": stability,
             "quality": quality, "geometry": geometry}
    score = sum(weights.get(k, 0.0) * parts[k] for k in parts)
    return clamp(score)


def final_text(text, char_conf, score):
    if score < SCORE_THRESHOLD or not text:
        return None, "unreadable"
    return "".join(c if cc >= CHAR_THRESHOLD else "?" for c, cc in zip(text, char_conf)), "ok"


def process_image(image):
    plate, bbox, det_conf, backend = detect_plate(image)
    if plate is None:
        return None
    quality = check_quality(plate)
    rectified = rectify(plate)
    text, char_conf, readings = read_plate(plate, rectified, quality)
    ocr_conf = float(np.mean([r["conf"] for r in readings])) if readings else 0.0
    return {
        "plate": plate, "rectified": rectified, "bbox": bbox, "det_conf": det_conf,
        "backend": backend, "quality": quality, "text": text, "char_conf": char_conf,
        "ocr_conf": ocr_conf, "readings": readings,
        "weight": max(quality["combined"], 0.05),
    }


def process_group(images):
    """images: list of BGR frames (photos or video frames) of the same
    vehicle. Detects and reads each one, then fuses the readings the same
    way multiple OCR views of a single photo are fused above."""
    results = [process_image(img) for img in images]
    results = [r for r in results if r is not None]
    if not results:
        return None

    per_image = [{"text": r["text"], "conf": r["ocr_conf"], "weight": r["weight"]} for r in results]
    fused_text, char_conf = fuse(per_image) if len(results) > 1 else (results[0]["text"], results[0]["char_conf"])
    fused_text = repair_format(fused_text)

    agreement = float(np.mean([
        (sum(a == b for a, b in zip(r["text"][:len(fused_text)], fused_text)) / len(fused_text))
        if fused_text else 0.0
        for r in results
    ])) if fused_text else 0.0
    stability = float(np.mean(char_conf)) if char_conf else 0.0
    ocr_conf = float(np.mean([r["ocr_conf"] for r in results]))
    quality = float(np.mean([r["quality"]["combined"] for r in results]))
    geometry = float(np.mean([r["quality"]["perspective"] for r in results]))

    score = confidence_score(ocr_conf, agreement, stability, quality, geometry)
    score = clamp(score + indian_format_bonus(fused_text))
    text, status = final_text(fused_text, char_conf, score)

    return {"results": results, "fused_text": fused_text, "score": score, "text": text,
            "status": status, "agreement": agreement}


# ---------------------------------------------------------------------------
# input helpers
# ---------------------------------------------------------------------------

def read_image_file(uploaded_file):
    """Decodes jpg, png, webp and avif alike through Pillow, since
    cv2.imdecode does not understand webp or avif on its own."""
    image = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _b64_thumb(image, height=96):
    h, w = image.shape[:2]
    scale = height / float(h)
    small = cv2.resize(image, (max(1, int(w * scale)), height))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode("ascii") if ok else ""


def read_video_frames(uploaded_file, max_frames=8):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    frames = []
    try:
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
        step = max(1, total // max_frames)
        idx = 0
        while cap.isOpened() and len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append(frame)
            idx += 1
        cap.release()
    finally:
        os.unlink(tmp_path)
    return frames


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# enhanced single-image Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SoApp | Number Plate Recognition",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root {
    --bg: #070b12;
    --panel: #0c131f;
    --panel-2: #101a29;
    --border: #24334a;
    --muted: #8290a5;
    --text: #edf3fb;
    --green: #2ee88b;
    --green-dark: #123d2b;
    --blue: #5aa7ff;
}

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(55, 122, 255, .09), transparent 28%),
        radial-gradient(circle at 92% 8%, rgba(46, 232, 139, .07), transparent 25%),
        var(--bg);
    color: var(--text);
}

.block-container {
    max-width: 1280px;
    padding: 2.2rem 2rem 4rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.hero {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 20px;
    margin-bottom: 1.5rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: 13px;
}

.brand-icon {
    width: 46px;
    height: 46px;
    display: grid;
    place-items: center;
    border-radius: 13px;
    background: linear-gradient(145deg, #14372a, #0d2019);
    border: 1px solid #245a43;
    font-size: 1.45rem;
}

.title {
    font-size: 2rem;
    line-height: 1.1;
    font-weight: 750;
    letter-spacing: -.035em;
}

.subtitle {
    color: var(--muted);
    font-size: .92rem;
    margin-top: 5px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 8px 12px;
    border: 1px solid #234a3a;
    border-radius: 999px;
    background: #0b1914;
    color: #74e7ae;
    font-size: .78rem;
    white-space: nowrap;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 10px rgba(46,232,139,.7);
}

.section-label {
    color: #8e9bb0;
    font-size: .72rem;
    letter-spacing: 1.7px;
    text-transform: uppercase;
    font-weight: 700;
    margin: 1.2rem 0 .55rem;
}

.upload-card {
    border: 1px solid var(--border);
    border-radius: 16px;
    background: linear-gradient(180deg, rgba(15,25,40,.96), rgba(9,16,27,.96));
    padding: 10px;
    box-shadow: 0 16px 45px rgba(0,0,0,.16);
}

div[data-testid="stFileUploader"] {
    border: 1px dashed #38506e;
    border-radius: 12px;
    background: rgba(7, 13, 22, .7);
    padding: .45rem;
}

div[data-testid="stFileUploaderDropzoneInstructions"] > div:first-child {
    color: #e8eef7;
}

div[data-testid="stFileUploaderDropzone"] {
    min-height: 145px;
}

div[data-testid="stFileUploader"] small {
    color: #73839a;
}

.image-card {
    border: 1px solid var(--border);
    border-radius: 14px;
    background: var(--panel);
    padding: 10px;
}

.image-title {
    color: #9aa8bc;
    font-size: .72rem;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    font-weight: 700;
    padding: 3px 5px 10px;
}

.result-shell {
    margin-top: 1.4rem;
    border: 1px solid var(--border);
    border-radius: 18px;
    background: linear-gradient(180deg, #0e1725, #09111d);
    overflow: hidden;
}

.result-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 17px 20px;
    border-bottom: 1px solid #1c2a3e;
}

.result-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
}

.check {
    width: 29px;
    height: 29px;
    border-radius: 50%;
    display: grid;
    place-items: center;
    background: var(--green);
    color: #06150d;
    font-weight: 900;
}

.result-state {
    color: #72e5ab;
    font-size: .75rem;
    border: 1px solid #24543e;
    background: #0b1d15;
    border-radius: 999px;
    padding: 5px 9px;
}

.plate-wrap {
    padding: 22px 20px 20px;
    text-align: center;
}

.eyebrow {
    color: #7e8da4;
    font-size: .69rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 700;
}

.plate-number {
    color: #3bea95;
    font: 750 clamp(1.9rem, 5vw, 3.15rem)/1.15 "Courier New", monospace;
    letter-spacing: .16em;
    margin: 9px 0 7px;
    word-break: break-word;
    text-shadow: 0 0 24px rgba(46,232,139,.12);
}

.plate-sub {
    color: #738198;
    font-size: .76rem;
}

.confidence-track {
    height: 7px;
    background: #182538;
    border-radius: 99px;
    overflow: hidden;
    margin: 17px auto 0;
    max-width: 500px;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #27c97a, #55efa5);
    border-radius: inherit;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
    padding: 0 20px 20px;
}

.metric {
    min-height: 92px;
    border: 1px solid #223148;
    border-radius: 12px;
    background: #0a121e;
    padding: 13px 14px;
}

.metric-label {
    color: #78869b;
    font-size: .68rem;
    text-transform: uppercase;
    letter-spacing: .8px;
    font-weight: 650;
}

.metric-value {
    color: #edf3fb;
    font-size: 1.3rem;
    font-weight: 750;
    margin-top: 7px;
}

.metric-value.good {
    color: #45e79a;
}

.metric-note {
    color: #596980;
    font-size: .67rem;
    margin-top: 3px;
}

.detail-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    padding: 0 20px 20px;
}

.detail {
    border-top: 1px solid #1c293c;
    padding: 12px 3px 0;
}

.detail-name {
    color: #748299;
    font-size: .7rem;
}

.detail-value {
    color: #cbd6e5;
    font-size: .83rem;
    margin-top: 4px;
    font-weight: 600;
}

.warning-box {
    margin-top: 1rem;
    border: 1px solid #584b28;
    background: #19160c;
    color: #d8c88c;
    border-radius: 12px;
    padding: 12px 14px;
    font-size: .82rem;
}

.error-box {
    border: 1px solid #5b2e36;
    background: #1b0e12;
    color: #ef9da9;
    border-radius: 13px;
    padding: 16px;
}

.footer-note {
    color: #56657a;
    font-size: .7rem;
    text-align: center;
    margin-top: 2rem;
}

@media (max-width: 900px) {
    .metric-grid { grid-template-columns: repeat(2, 1fr); }
    .detail-grid { grid-template-columns: 1fr; }
    .hero { align-items: flex-start; }
}

@media (max-width: 600px) {
    .block-container { padding: 1.3rem 1rem 3rem; }
    .title { font-size: 1.55rem; }
    .status-pill { display: none; }
    .metric-grid { grid-template-columns: repeat(2, 1fr); padding: 0 12px 12px; }
    .result-head, .plate-wrap { padding-left: 14px; padding-right: 14px; }
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="hero">
  <div>
    <div class="brand">
      <div class="brand-icon">🚘</div>
      <div>
        <div class="title">Number Plate Recognition</div>
        <div class="subtitle">AI-powered Indian vehicle number plate detection & OCR</div>
      </div>
    </div>
  </div>
  <div class="status-pill"><span class="status-dot"></span>Recognition system ready</div>
</div>
""", unsafe_allow_html=True)

# Single-image input only.
st.markdown('<div class="section-label">Upload vehicle image</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload one clear vehicle image",
    type=["jpg", "jpeg", "png", "webp", "avif"],
    accept_multiple_files=False,
    help="Upload a JPG, JPEG, PNG, WEBP or AVIF image. Only one image can be processed at a time.",
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded is None:
    st.markdown("""
    <div style="text-align:center;color:#617088;font-size:.78rem;margin-top:18px">
        Upload an image to automatically detect, clean and read the number plate.
    </div>
    """, unsafe_allow_html=True)
else:
    image = read_image_file(uploaded)

    with st.spinner("Detecting plate and running OCR..."):
        result = process_group([image])

    if result is None:
        st.markdown(
            '<div class="error-box">⚠️ No license plate was detected. '
            'Try a closer, brighter or less obstructed vehicle image.</div>',
            unsafe_allow_html=True
        )
        st.markdown('<div class="section-label">Uploaded image</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        best = max(result["results"], key=lambda r: r["quality"]["combined"])
        annotated = image.copy()
        x1, y1, x2, y2 = best["bbox"]

        # More polished annotation: bounding box + readable label.
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 230, 125), 4)
        label = result["text"] or "PLATE DETECTED"
        confidence_label = f"{label}  {result['score']:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.55, min(1.15, image.shape[1] / 1200))
        thickness = max(2, int(scale * 2.5))
        (lw, lh), baseline = cv2.getTextSize(confidence_label, font, scale, thickness)
        label_y = y1 - 10
        if label_y - lh - baseline < 0:
            label_y = y2 + lh + baseline + 10
        box_top = label_y - lh - baseline - 10
        box_bottom = label_y + 5
        box_right = min(image.shape[1] - 1, x1 + lw + 16)
        cv2.rectangle(annotated, (x1, box_top), (box_right, box_bottom), (0, 185, 95), -1)
        cv2.putText(
            annotated, confidence_label, (x1 + 8, label_y - 4),
            font, scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

        # Top image section: exactly the three views requested.
        st.markdown('<div class="section-label">Recognition preview</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap="medium")

        with c1:
            st.markdown('<div class="image-card"><div class="image-title">01 · Uploaded image</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            cleaned = best["rectified"] if best["rectified"] is not None else best["plate"]
            disp = (
                cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
                if cleaned.ndim == 2
                else cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)
            )
            st.markdown('<div class="image-card"><div class="image-title">02 · Detected plate · cleaned</div>', unsafe_allow_html=True)
            st.image(disp, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="image-card"><div class="image-title">03 · Annotated result</div>', unsafe_allow_html=True)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Main result card.
        display_text = result["text"] or result["fused_text"] or "UNREADABLE"
        score_pct = result["score"] * 100
        confidence_word = (
            "Very High" if score_pct >= 85 else
            "High" if score_pct >= 70 else
            "Moderate" if score_pct >= 50 else
            "Low"
        )

        st.markdown(f"""
        <div class="result-shell">
          <div class="result-head">
            <div class="result-title"><span class="check">✓</span> Recognition Result</div>
            <div class="result-state">{confidence_word} confidence</div>
          </div>

          <div class="plate-wrap">
            <div class="eyebrow">Detected plate number</div>
            <div class="plate-number">{display_text}</div>
            <div class="plate-sub">OCR reading • Indian vehicle plate format</div>
            <div class="confidence-track">
              <div class="confidence-fill" style="width:{score_pct:.1f}%"></div>
            </div>
          </div>

          <div class="metric-grid">
            <div class="metric">
              <div class="metric-label">Overall Accuracy</div>
              <div class="metric-value good">{score_pct:.1f}%</div>
              <div class="metric-note">combined score</div>
            </div>
            <div class="metric">
              <div class="metric-label">Detection</div>
              <div class="metric-value">{best["det_conf"]:.1%}</div>
              <div class="metric-note">plate localization</div>
            </div>
            <div class="metric">
              <div class="metric-label">OCR Confidence</div>
              <div class="metric-value">{best["ocr_conf"]:.1%}</div>
              <div class="metric-note">text recognition</div>
            </div>
            <div class="metric">
              <div class="metric-label">Image Quality</div>
              <div class="metric-value">{best["quality"]["combined"]:.1%}</div>
              <div class="metric-note">blur / exposure / noise</div>
            </div>
            <div class="metric">
              <div class="metric-label">Agreement</div>
              <div class="metric-value">{result["agreement"]:.1%}</div>
              <div class="metric-note">OCR consistency</div>
            </div>
          </div>

          <div class="detail-grid">
            <div class="detail">
              <div class="detail-name">Confidence strength</div>
              <div class="detail-value">{confidence_word}</div>
            </div>
            <div class="detail">
              <div class="detail-name">Detection engine</div>
              <div class="detail-value">{best["backend"].upper()}</div>
            </div>
            <div class="detail">
              <div class="detail-name">Plate geometry</div>
              <div class="detail-value">{best["quality"]["perspective"]:.1%} quality</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if not result["text"]:
            st.markdown(
                '<div class="warning-box">⚠️ A plate was detected, but the OCR reading '
                'did not pass the reliability threshold. The displayed text may contain uncertain characters.</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="footer-note">Detection → perspective correction → image enhancement → OCR → confidence fusion</div>',
            unsafe_allow_html=True
        )
