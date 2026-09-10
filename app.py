import os
import json
import html
import itertools
from io import BytesIO

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image

try:
    import pillow_avif  # noqa: F401
except Exception:
    pass

try:
    from rapidocr import RapidOCR
    RAPIDOCR_AVAILABLE = True
except Exception as e:
    RapidOCR = None
    RAPIDOCR_AVAILABLE = False
    print(f"RapidOCR unavailable: {e}")

from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "plate_detector.pt")
CONFIG_PATH = os.path.join(BASE_DIR, "pipeline_config.json")

DEFAULT_CONFIG = {
    "plate_size": [256, 96],
    "quality_weights": {"blur": 0.3, "exposure": 0.15, "noise": 0.15, "resolution": 0.2, "perspective": 0.1, "occlusion": 0.1},
    "score_weights": {"ocr": 0.3, "agreement": 0.2, "stability": 0.2, "quality": 0.15, "geometry": 0.15},
    "char_threshold": 0.35,
    "score_threshold": 0.25,
    "detector_trained": True,
}

try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
except Exception:
    CONFIG = DEFAULT_CONFIG

PLATE_SIZE = tuple(CONFIG.get("plate_size", [256, 96]))
QUALITY_WEIGHTS = CONFIG.get("quality_weights", DEFAULT_CONFIG["quality_weights"])
SCORE_WEIGHTS = CONFIG.get("score_weights", DEFAULT_CONFIG["score_weights"])

INDIAN_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
}

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"


@st.cache_resource(show_spinner=False)
def get_detector():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Could not find {MODEL_PATH}")
    return YOLO(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_rapidocr():
    if not RAPIDOCR_AVAILABLE:
        return None
    try:
        return RapidOCR()
    except Exception as e:
        print(f"RapidOCR initialization failed: {e}")
        return None


detector = get_detector()


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(v)))


def normalize_text(text):
    if text is None:
        return ""
    text = str(text).upper()
    return "".join(c for c in text if c in LETTERS + DIGITS)


def blur_score(image):
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return clamp(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0)


def exposure_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return clamp(1.0 - abs(float(np.mean(gray)) - 127.5) / 127.5)


def noise_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    residual = cv2.absdiff(gray, cv2.GaussianBlur(gray, (3, 3), 0))
    return clamp(1.0 - float(np.mean(residual)) / 50.0)


def resolution_score(image):
    h, w = image.shape[:2]
    return clamp((w * h) / 15000.0)


def perspective_score(image):
    h, w = image.shape[:2]
    if not h:
        return 0.0
    r = w / float(h)
    if 2.0 <= r <= 5.5:
        return 1.0
    if 1.5 <= r < 2.0:
        return 0.7
    if r > 8.0:
        return 0.5
    return 0.4


def occlusion_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return clamp(float(np.std(gray)) / 80.0)


def calculate_quality_score(plate):
    scores = {
        "blur": blur_score(plate), "exposure": exposure_score(plate), "noise": noise_score(plate),
        "resolution": resolution_score(plate), "perspective": perspective_score(plate), "occlusion": occlusion_score(plate)
    }
    total = sum(scores[k] * QUALITY_WEIGHTS.get(k, 0.0) for k in scores)
    weights = sum(QUALITY_WEIGHTS.get(k, 0.0) for k in scores)
    return (clamp(total / weights) if weights else 0.0), scores


def order_quad(points):
    points = np.asarray(points, dtype=np.float32)
    s = points.sum(axis=1)
    d = np.diff(points, axis=1).reshape(-1)
    return np.array([points[np.argmin(s)], points[np.argmin(d)], points[np.argmax(s)], points[np.argmax(d)]], dtype=np.float32)


def find_plate_quad(plate):
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) if plate.ndim == 3 else plate.copy()
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 140)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    area_total = float(h * w)
    best, best_score = None, -1
    for c in contours:
        area = cv2.contourArea(c)
        if area < area_total * 0.18 or area > area_total * 0.98:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        for eps in (0.02, 0.03, 0.04):
            a = cv2.approxPolyDP(c, eps * peri, True)
            if len(a) != 4:
                continue
            q = order_quad(a.reshape(4, 2))
            wt = np.linalg.norm(q[1] - q[0]); wb = np.linalg.norm(q[2] - q[3])
            hl = np.linalg.norm(q[3] - q[0]); hr = np.linalg.norm(q[2] - q[1])
            ww, hh = (wt + wb) / 2, (hl + hr) / 2
            if hh <= 1 or ww <= 1:
                continue
            aspect = ww / hh
            if not 2.0 <= aspect <= 8.0:
                continue
            rectangularity = area / max(ww * hh, 1.0)
            score = (area / area_total) * 0.65 + rectangularity * 0.35
            if score > best_score:
                best_score, best = score, q
    return best


def rectify_plate(plate):
    if plate is None or plate.size == 0:
        return None
    tw, th = PLATE_SIZE
    quad = find_plate_quad(plate)
    if quad is not None:
        dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(quad, dst)
        return cv2.warpPerspective(plate, M, (tw, th), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    padded = cv2.copyMakeBorder(plate, 8, 8, 12, 12, cv2.BORDER_REPLICATE)
    return cv2.resize(padded, (tw, th), interpolation=cv2.INTER_CUBIC)


def ocr_variants(image):
    """Small set of faithful variants; avoid aggressive thresholding on blur."""
    if image is None or image.size == 0:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    h, w = gray.shape[:2]
    gray = cv2.copyMakeBorder(gray, max(4, h // 20), max(4, h // 20), max(6, w // 50), max(6, w // 50), cv2.BORDER_REPLICATE)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8)).apply(gray)
    denoise = cv2.bilateralFilter(clahe, 5, 25, 25)
    soft = cv2.GaussianBlur(denoise, (0, 0), 1.0)
    sharp = cv2.addWeighted(denoise, 1.25, soft, -0.25, 0)
    normalized = cv2.divide(clahe, cv2.GaussianBlur(clahe, (0, 0), 15), scale=180)
    return [("gray", gray), ("clahe", clahe), ("denoise", denoise), ("sharp", sharp), ("normalized", normalized)]


def clean_candidate(text, min_len=1, max_len=12):
    text = normalize_text(text)
    return text if min_len <= len(text) <= max_len else ""


def rapid_read(image):
    engine = get_rapidocr()
    if engine is None:
        return []
    out = []
    for mode in ("full", "rec"):
        try:
            if mode == "full":
                result = engine(image)
            else:
                result = engine(image, use_det=False, use_cls=False, use_rec=True)
            texts = getattr(result, "txts", None) or []
            scores = getattr(result, "scores", None) or []
            for i, t in enumerate(texts):
                t = clean_candidate(t, 1, 12)
                if t:
                    out.append((t, float(scores[i]) if i < len(scores) else 0.0, f"rapid-{mode}"))
        except Exception as e:
            print(f"RapidOCR {mode} failed: {e}")
    return out


def tess_read(image, whitelist, psms=(7, 8, 10, 13)):
    out = []
    for psm in psms:
        cfg = f"--psm {psm} -l eng -c tessedit_char_whitelist={whitelist}"
        try:
            text = clean_candidate(pytesseract.image_to_string(image, config=cfg), 1, 12)
            if not text:
                continue
            confs = []
            try:
                d = pytesseract.image_to_data(image, config=cfg, output_type=pytesseract.Output.DICT)
                for c in d.get("conf", []):
                    try:
                        v = float(c)
                        if v >= 0:
                            confs.append(v / 100.0)
                    except Exception:
                        pass
            except Exception:
                pass
            out.append((text, float(np.mean(confs)) if confs else 0.30, f"tess-{psm}"))
        except Exception:
            pass
    return out


def group_candidates(group_image, whitelist, expected_len):
    """Recognize a plate group independently. This prevents one bad character from corrupting the whole line."""
    variants = ocr_variants(group_image)
    candidates = []
    for vname, img in variants[:4]:
        for text, conf, src in rapid_read(img):
            if len(text) == expected_len and all(c in whitelist for c in text):
                candidates.append((text, conf, f"{vname}-{src}"))
        for text, conf, src in tess_read(img, whitelist, (7, 8, 10, 13)):
            if len(text) == expected_len and all(c in whitelist for c in text):
                candidates.append((text, conf, f"{vname}-{src}"))
    # exact candidate consensus
    grouped = {}
    for text, conf, src in candidates:
        x = grouped.setdefault(text, {"scores": [], "sources": []})
        x["scores"].append(conf); x["sources"].append(src)
    ranked = []
    for text, d in grouped.items():
        mean = float(np.mean(d["scores"]))
        count = len(d["scores"])
        ranked.append((text, mean + min(count, 4) * 0.08, mean, count))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:8]


def whole_candidates(plate, rectified):
    sources = []
    for name, img in (("raw", plate), ("rectified", rectified)):
        if img is None:
            continue
        for vname, variant in ocr_variants(img):
            sources.append((f"{name}-{vname}", variant))
    results = []
    for name, img in sources:
        for text, conf, src in rapid_read(img):
            if len(text) <= 12:
                results.append((text, conf, f"{name}-{src}"))
        for text, conf, src in tess_read(img, LETTERS + DIGITS, (6, 7, 8, 11, 13)):
            if len(text) <= 12:
                results.append((text, conf, f"{name}-{src}"))
    return results


def extract_groups(image):
    """Use overlapping group crops so blur or spacing does not fall exactly on a boundary."""
    if image is None:
        return []
    h, w = image.shape[:2]
    # Text generally occupies the middle of the detected plate.
    y1, y2 = int(h * 0.10), int(h * 0.92)
    base = image[y1:y2, :]
    w = base.shape[1]
    specs = [(0.00, 0.25, LETTERS, 2), (0.20, 0.43, DIGITS, 2), (0.38, 0.63, LETTERS, 2), (0.58, 1.00, DIGITS, 4)]
    groups = []
    for a, b, whitelist, n in specs:
        x1 = max(0, int(w * a) - max(2, int(w * 0.015)))
        x2 = min(w, int(w * b) + max(2, int(w * 0.015)))
        groups.append((base[:, x1:x2], whitelist, n))
    return groups


def char_candidates(image, whitelist):
    """Per-character fallback using equal-width cells, useful when whole-line OCR drops a leading letter."""
    if image is None or image.size == 0:
        return []
    h, w = image.shape[:2]
    results = []
    for i in range(2):
        x1 = max(0, int(w * i / 2) - 2)
        x2 = min(w, int(w * (i + 1) / 2) + 2)
        cell = image[:, x1:x2]
        chars = []
        for vname, v in ocr_variants(cell)[:3]:
            for t, c, src in tess_read(v, whitelist, (10, 13)):
                if t and t[0] in whitelist:
                    chars.append((t[0], c, f"{vname}-{src}"))
        if chars:
            chars.sort(key=lambda x: x[1], reverse=True)
            results.append(chars[:5])
        else:
            results.append([])
    return results


def build_structured_candidates(plate, rectified):
    """Decode AA00AA0000 from independent group evidence, without hard-coding a plate."""
    base = rectified if rectified is not None else plate
    group_specs = extract_groups(base)
    if len(group_specs) != 4:
        return []
    group_ranked = []
    for crop, whitelist, n in group_specs:
        group_ranked.append(group_candidates(crop, whitelist, n))

    # If a group has no complete reading, retain no synthetic guess.
    if any(not g for g in group_ranked):
        return []

    combinations = []
    top_each = [g[:5] for g in group_ranked]
    for combo in itertools.product(*top_each):
        text = "".join(x[0] for x in combo)
        if len(text) != 10:
            continue
        if text[:2] not in INDIAN_STATE_CODES:
            state_bonus = 0.0
        else:
            state_bonus = 0.45
        score = state_bonus + sum(x[1] for x in combo) / 4.0
        combinations.append((text, score, combo))
    combinations.sort(key=lambda x: x[1], reverse=True)
    return combinations[:12]


def candidate_score(text, source_conf=0.0, repetition=0):
    text = normalize_text(text)
    if len(text) != 10:
        return 0.05 * clamp(source_conf)
    state = text[:2] in INDIAN_STATE_CODES
    structure = (0.45 * state + 0.20 * text[2:4].isdigit() + 0.15 * text[4:6].isalpha() + 0.20 * text[6:10].isdigit())
    return 0.68 * structure + 0.22 * clamp(source_conf) + 0.10 * min(repetition / 4.0, 1.0)


def run_ocr(plate, rectified):
    all_results = []
    for text, conf, src in whole_candidates(plate, rectified):
        all_results.append({"text": text, "confidence": conf * 100, "source": src, "image": rectified if rectified is not None else plate})

    # Structured group decoding is given strong weight only when every group has
    # a real OCR reading. It is not character invention.
    structured = build_structured_candidates(plate, rectified)
    for text, score, combo in structured:
        all_results.append({"text": text, "confidence": clamp(score / 1.6) * 100, "source": "structured-groups", "image": rectified if rectified is not None else plate, "structured": True})

    if not all_results:
        return [], None

    grouped = {}
    for r in all_results:
        t = normalize_text(r["text"])
        if not t:
            continue
        g = grouped.setdefault(t, {"items": [], "count": 0})
        g["items"].append(r); g["count"] += 1

    best = None
    best_s = -1
    for text, g in grouped.items():
        mean_conf = float(np.mean([clamp(x["confidence"] / 100) for x in g["items"]]))
        s = candidate_score(text, mean_conf, g["count"])
        # Valid state code is mandatory for a complete normal plate.
        if len(text) == 10 and text[:2] in INDIAN_STATE_CODES:
            s += 0.14
        if len(text) == 10 and text[:2] not in INDIAN_STATE_CODES:
            s -= 0.18
        if any(x.get("structured") for x in g["items"]):
            s += 0.08
        if s > best_s:
            best_s = s
            best = {**max(g["items"], key=lambda x: x["confidence"]), "text": text, "agreement": min(g["count"] / 4.0, 1.0), "combined_score": clamp(s)}
    return all_results, best


def detect_plate(image):
    try:
        results = detector.predict(source=image, verbose=False)
    except Exception as e:
        print(f"Detection error: {e}")
        return None, None, 0.0
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None, None, 0.0
    boxes = results[0].boxes
    confs = [float(b.conf[0].item()) for b in boxes]
    i = int(np.argmax(confs))
    b = boxes[i]
    x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
    h, w = image.shape[:2]
    x1, y1 = max(0, min(x1, w - 1)), max(0, min(y1, h - 1))
    x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None, None, confs[i]
    return image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2), confs[i]


def main():
    st.set_page_config(page_title="License Plate Recognition", page_icon="🚗", layout="wide")
    st.markdown("""
    <style>
    .stApp{background:#080d16;color:#e8edf5}.block-container{max-width:1240px;padding-top:2rem}
    .title{font-size:2.1rem;font-weight:700}.subtitle{color:#8e9bb0;margin-bottom:1.2rem}
    .result-box{border:1px solid #26364d;border-radius:12px;background:#0d1624;padding:20px;text-align:center}
    .plate{color:#35e58d;font:700 2.2rem 'Courier New',monospace;letter-spacing:3px}
    .metric{border:1px solid #26364d;border-radius:10px;background:#0a111c;padding:15px}.metric-label{color:#8190a7;font-size:.78rem}.metric-value{font-size:1.35rem;font-weight:700;margin-top:4px}
    .candidate{color:#aab6c8;font-family:monospace;font-size:.85rem;margin-right:8px}
    </style>""", unsafe_allow_html=True)
    st.markdown('<div class="title">License Plate Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Detect and read Indian vehicle license plates using YOLO, OpenCV, RapidOCR and Tesseract.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload a vehicle image", type=["jpg","jpeg","png","webp","avif"], help="The pipeline uses multiple OCR views for clear, low-resolution and blurred plates.")
    if uploaded is None:
        st.info("Upload a vehicle image to start recognition.")
        return
    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Could not decode the uploaded image.")
        return
    if not st.button("Recognize Plate", type="primary", use_container_width=True):
        return
    with st.spinner("Detecting plate and running robust OCR..."):
        original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        plate, bbox, detector_conf = detect_plate(image_bgr)
        if plate is None:
            st.image(original_rgb, caption="Uploaded Image", use_container_width=True)
            st.error("No license plate detected.")
            return
        quality, _ = calculate_quality_score(plate)
        rectified = rectify_plate(plate)
        results, best = run_ocr(plate, rectified)
        if best:
            text = normalize_text(best["text"])
            ocr_conf = clamp(float(best.get("confidence", 0)) / 100)
            agreement = float(best.get("agreement", 0))
            cleaned = best.get("image", rectified)
        else:
            text, ocr_conf, agreement, cleaned = "", 0.0, 0.0, rectified
        geometry = perspective_score(plate)
        overall = clamp(
            SCORE_WEIGHTS.get("ocr", .3) * ocr_conf +
            SCORE_WEIGHTS.get("agreement", .2) * agreement +
            SCORE_WEIGHTS.get("stability", .2) * agreement +
            SCORE_WEIGHTS.get("quality", .15) * quality +
            SCORE_WEIGHTS.get("geometry", .15) * geometry
        )
        annotated = image_bgr.copy()
        x1,y1,x2,y2 = bbox
        cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),3)
        label = f"{text}  {detector_conf:.0%}" if text else "PLATE DETECTED"
        (lw,lh),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.7,2)
        ly=max(30,y1)
        cv2.rectangle(annotated,(x1,ly-lh-15),(x1+lw+12,ly),(0,180,0),-1)
        cv2.putText(annotated,label,(x1+6,ly-8),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
        annotated_rgb=cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB)
    c1,c2,c3=st.columns(3)
    with c1: st.image(original_rgb,caption="Uploaded Image",use_container_width=True)
    with c2:
        if cleaned is not None:
            cd=cv2.cvtColor(cleaned,cv2.COLOR_GRAY2RGB) if cleaned.ndim==2 else cv2.cvtColor(cleaned,cv2.COLOR_BGR2RGB)
            st.image(cd,caption="Detected Plate (Cleaned)",use_container_width=True)
    with c3: st.image(annotated_rgb,caption="Annotated Image",use_container_width=True)
    if text:
        st.markdown(f'<div class="result-box"><div style="color:#8090a8;font-size:.75rem;letter-spacing:1px">DETECTED PLATE NUMBER</div><div class="plate">{html.escape(text)}</div></div>',unsafe_allow_html=True)
        st.write("")
        cols=st.columns(4)
        for col,label,value in zip(cols,["Overall Confidence","Detection Confidence","OCR Confidence","Image Quality"],[overall,detector_conf,ocr_conf,quality]):
            with col: st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value:.1%}</div></div>',unsafe_allow_html=True)
        candidates=[]
        for r in results:
            t=normalize_text(r.get("text",""))
            if t and t not in candidates: candidates.append(t)
        if candidates:
            st.markdown("**OCR candidates:** " + " ".join(f'<span class="candidate">{html.escape(x)}</span>' for x in candidates[:12]),unsafe_allow_html=True)
    else:
        st.warning("Plate detected, but OCR could not produce a reliable reading.")


if __name__ == "__main__":
    main()
