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
    "quality_weights": {
        "blur": 0.30,
        "exposure": 0.15,
        "noise": 0.15,
        "resolution": 0.20,
        "perspective": 0.10,
        "occlusion": 0.10,
    },
    "score_weights": {
        "ocr": 0.30,
        "agreement": 0.20,
        "stability": 0.20,
        "quality": 0.15,
        "geometry": 0.15,
    },
    "char_threshold": 0.35,
    "score_threshold": 0.25,
}


try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        CONFIG = {**DEFAULT_CONFIG, **json.load(f)}
except Exception:
    CONFIG = DEFAULT_CONFIG


PLATE_SIZE = tuple(CONFIG.get("plate_size", [256, 96]))
QUALITY_WEIGHTS = CONFIG.get(
    "quality_weights",
    DEFAULT_CONFIG["quality_weights"]
)
SCORE_WEIGHTS = CONFIG.get(
    "score_weights",
    DEFAULT_CONFIG["score_weights"]
)
CHAR_THRESHOLD = CONFIG.get("char_threshold", 0.35)
SCORE_THRESHOLD = CONFIG.get("score_threshold", 0.25)


CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

SIMILAR = {
    "O": ["0"],
    "0": ["O"],
    "I": ["1"],
    "1": ["I"],
    "B": ["8"],
    "8": ["B"],
    "S": ["5"],
    "5": ["S"],
    "Z": ["2"],
    "2": ["Z"],
    "G": ["6"],
    "6": ["G"],
}


INDIAN_STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA",
    "GJ", "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
    "TN", "TR", "TS", "UK", "UP", "WB",
}


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(v)))


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).upper()

    return "".join(
        c for c in text
        if c in CHARS
    )


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def get_detector():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Could not find trained detector: {MODEL_PATH}"
        )

    return YOLO(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def get_haar():

    path = os.path.join(
        getattr(cv2.data, "haarcascades", ""),
        "haarcascade_russian_plate_number.xml"
    )

    if os.path.isfile(path) and hasattr(
        cv2,
        "CascadeClassifier"
    ):

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


# ============================================================
# PLATE DETECTION
# ============================================================

def _contour_guess(image):

    h, w = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.dilate(
        cv2.Canny(gray, 50, 150),
        np.ones((3, 3), np.uint8)
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_area = -1

    for c in contours:

        x, y, cw, ch = cv2.boundingRect(c)

        if cw < 25 or ch < 10:
            continue

        ratio = cw / ch

        if not (1.3 <= ratio <= 8.0):
            continue

        if cw * ch > best_area:

            best_area = cw * ch

            best = (
                x,
                y,
                x + cw,
                y + ch
            )

    return best


def detect_plate(image):

    h, w = image.shape[:2]

    for conf, imgsz in (
        (0.25, 640),
        (0.10, 1280)
    ):

        try:

            results = detector.predict(
                source=image,
                conf=conf,
                imgsz=imgsz,
                verbose=False
            )

        except Exception as e:

            print(f"detector error: {e}")
            results = None

        if (
            results
            and results[0].boxes is not None
            and len(results[0].boxes) > 0
        ):

            boxes = results[0].boxes

            confs = [
                float(
                    b.conf[0].item()
                )
                for b in boxes
            ]

            i = int(
                np.argmax(confs)
            )

            x1, y1, x2, y2 = map(
                int,
                boxes[i].xyxy[0]
                .cpu()
                .numpy()
            )

            x1 = max(
                0,
                min(x1, w - 1)
            )

            y1 = max(
                0,
                min(y1, h - 1)
            )

            x2 = max(
                0,
                min(x2, w)
            )

            y2 = max(
                0,
                min(y2, h)
            )

            if x2 > x1 and y2 > y1:

                return (
                    image[y1:y2, x1:x2].copy(),
                    (x1, y1, x2, y2),
                    confs[i],
                    "yolo"
                )


    # Haar fallback
    if haar is not None:

        boxes = haar.detectMultiScale(
            cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            ),
            1.1,
            4
        )

        if len(boxes) > 0:

            x, y, cw, ch = max(
                boxes,
                key=lambda b: b[2] * b[3]
            )

            x1 = x
            y1 = y
            x2 = x + cw
            y2 = y + ch

            return (
                image[y1:y2, x1:x2].copy(),
                (x1, y1, x2, y2),
                0.35,
                "haar"
            )


    # Contour fallback
    guess = _contour_guess(image)

    if guess is not None:

        x1, y1, x2, y2 = guess

        return (
            image[y1:y2, x1:x2].copy(),
            (x1, y1, x2, y2),
            0.20,
            "contour"
        )


    return None, None, 0.0, "none"


# ============================================================
# IMAGE QUALITY
# ============================================================

def check_quality(image):

    gray = (
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
        if image.ndim == 3
        else image
    )

    h, w = gray.shape[:2]

    blur = clamp(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var() / 500.0
    )

    exposure = clamp(
        1.0 -
        abs(
            float(np.mean(gray)) - 127.5
        ) / 127.5
    )

    residual = cv2.absdiff(
        gray,
        cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )
    )

    noise = clamp(
        1.0 -
        float(np.mean(residual)) / 50.0
    )

    resolution = clamp(
        (w * h) / 15000.0
    )

    ratio = (
        w / float(h)
        if h
        else 0.0
    )

    if 2.0 <= ratio <= 5.5:
        perspective = 1.0

    elif 1.2 <= ratio < 2.0:
        perspective = 0.7

    elif ratio > 8.0:
        perspective = 0.5

    else:
        perspective = 0.4

    occlusion = clamp(
        float(np.std(gray)) / 80.0
    )

    scores = {
        "blur": blur,
        "exposure": exposure,
        "noise": noise,
        "resolution": resolution,
        "perspective": perspective,
        "occlusion": occlusion
    }

    total = sum(
        scores[k] *
        QUALITY_WEIGHTS.get(k, 0.0)
        for k in scores
    )

    weight_sum = sum(
        QUALITY_WEIGHTS.get(k, 0.0)
        for k in scores
    )

    scores["combined"] = (
        clamp(total / weight_sum)
        if weight_sum
        else 0.0
    )

    return scores


# ============================================================
# RECTIFICATION
# ============================================================

def _order_quad(points):

    points = np.asarray(
        points,
        dtype=np.float32
    )

    s = points.sum(axis=1)

    d = np.diff(
        points,
        axis=1
    ).reshape(-1)

    return np.array(
        [
            points[np.argmin(s)],
            points[np.argmin(d)],
            points[np.argmax(s)],
            points[np.argmax(d)]
        ],
        dtype=np.float32
    )


def _find_plate_quad(plate):

    gray = (
        cv2.cvtColor(
            plate,
            cv2.COLOR_BGR2GRAY
        )
        if plate.ndim == 3
        else plate.copy()
    )

    edges = cv2.Canny(
        cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        ),
        40,
        140
    )

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = gray.shape[:2]

    area_total = float(h * w)

    best = None
    best_score = -1.0

    for c in contours:

        area = cv2.contourArea(c)

        if (
            area < area_total * 0.18
            or area > area_total * 0.98
        ):
            continue

        peri = cv2.arcLength(
            c,
            True
        )

        if peri <= 0:
            continue

        for eps in (
            0.02,
            0.03,
            0.04
        ):

            approx = cv2.approxPolyDP(
                c,
                eps * peri,
                True
            )

            if len(approx) != 4:
                continue

            q = _order_quad(
                approx.reshape(
                    4,
                    2
                )
            )

            width = (
                np.linalg.norm(
                    q[1] - q[0]
                )
                +
                np.linalg.norm(
                    q[2] - q[3]
                )
            ) / 2

            height = (
                np.linalg.norm(
                    q[3] - q[0]
                )
                +
                np.linalg.norm(
                    q[2] - q[1]
                )
            ) / 2

            if height <= 1 or width <= 1:
                continue

            aspect = width / height

            if not (
                1.2 <= aspect <= 8.0
            ):
                continue

            rectangularity = (
                area /
                max(
                    width * height,
                    1.0
                )
            )

            score = (
                (area / area_total) * 0.65
                +
                rectangularity * 0.35
            )

            if score > best_score:

                best_score = score
                best = q

    return best


def rectify(plate):

    if (
        plate is None
        or plate.size == 0
    ):
        return None

    tw, th = PLATE_SIZE

    quad = _find_plate_quad(
        plate
    )

    if quad is not None:

        dst = np.array(
            [
                [0, 0],
                [tw - 1, 0],
                [tw - 1, th - 1],
                [0, th - 1]
            ],
            dtype=np.float32
        )

        matrix = cv2.getPerspectiveTransform(
            quad,
            dst
        )

        return cv2.warpPerspective(
            plate,
            matrix,
            (tw, th),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    padded = cv2.copyMakeBorder(
        plate,
        8,
        8,
        12,
        12,
        cv2.BORDER_REPLICATE
    )

    return cv2.resize(
        padded,
        (tw, th),
        interpolation=cv2.INTER_CUBIC
    )


# ============================================================
# OCR PREPROCESSING
# ============================================================

def ocr_variants(
    image,
    quality
):

    if (
        image is None
        or image.size == 0
    ):
        return []

    gray = (
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )
        if image.ndim == 3
        else image.copy()
    )

    h, w = gray.shape[:2]

    gray = cv2.copyMakeBorder(
        gray,
        max(4, h // 20),
        max(4, h // 20),
        max(6, w // 45),
        max(6, w // 45),
        cv2.BORDER_REPLICATE
    )

    scale = (
        6.0
        if quality["resolution"] < 0.4
        else 4.0
    )

    gray = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)

    _, otsu = cv2.threshold(
        clahe,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    variants = [
        ("gray", gray),
        ("otsu", otsu)
    ]

    worst_name, worst_score = min(
        (
            ("noise", quality["noise"]),
            ("blur", quality["blur"]),
            ("exposure", quality["exposure"])
        ),
        key=lambda kv: kv[1]
    )

    if worst_score < 0.6:

        if worst_name == "noise":

            variants.append(
                (
                    "denoise",
                    cv2.bilateralFilter(
                        clahe,
                        5,
                        30,
                        30
                    )
                )
            )

        elif worst_name == "blur":

            soft = cv2.GaussianBlur(
                clahe,
                (0, 0),
                1.2
            )

            variants.append(
                (
                    "sharp",
                    cv2.addWeighted(
                        clahe,
                        1.6,
                        soft,
                        -0.6,
                        0
                    )
                )
            )

        else:

            background = cv2.GaussianBlur(
                clahe,
                (0, 0),
                15
            )

            variants.append(
                (
                    "normalized",
                    cv2.divide(
                        clahe,
                        background,
                        scale=180
                    )
                )
            )

    else:

        variants.append(
            ("clahe", clahe)
        )

    return variants


def split_two_line(image):

    h, w = image.shape[:2]

    if (
        h == 0
        or w / float(h) > 2.3
    ):
        return None

    mid = h // 2

    pad = max(
        2,
        h // 12
    )

    top = image[
        :mid + pad,
        :
    ]

    bottom = image[
        max(0, mid - pad):,
        :
    ]

    return top, bottom


# ============================================================
# OCR ENGINES
# ============================================================

def _tesseract_read(image):

    out = []

    for psm in (
        7,
        8,
        6
    ):

        config = (
            f"--psm {psm} "
            f"-c tessedit_char_whitelist={CHARS}"
        )

        try:

            data = pytesseract.image_to_data(
                image,
                config=config,
                output_type=pytesseract.Output.DICT
            )

        except Exception:
            continue

        text = normalize_text(
            "".join(
                data.get(
                    "text",
                    []
                )
            )
        )

        if not text:
            continue

        values = [
            float(c)
            for c in data.get(
                "conf",
                []
            )
            if c not in (
                "-1",
                -1
            )
        ]

        conf = (
            float(np.mean(values))
            / 100.0
            if values
            else 0.3
        )

        out.append(
            (
                text,
                max(conf, 0.15)
            )
        )

    return out


def _rapidocr_read(image):

    engine = get_rapidocr()

    if engine is None:
        return []

    out = []

    try:

        result = engine(
            image,
            use_det=False,
            use_cls=False,
            use_rec=True
        )

    except Exception as e:

        print(
            f"rapidocr failed: {e}"
        )

        return out

    texts = (
        getattr(
            result,
            "txts",
            None
        )
        or []
    )

    scores = (
        getattr(
            result,
            "scores",
            None
        )
        or []
    )

    for i, text in enumerate(
        texts
    ):

        text = normalize_text(
            text
        )

        if text:

            conf = (
                float(scores[i])
                if i < len(scores)
                else 0.5
            )

            out.append(
                (
                    text,
                    max(conf, 0.15)
                )
            )

    return out


# ============================================================
# OCR FUSION
# ============================================================

def _align(
    ref,
    seq,
    gap=-1,
    match=2,
    mismatch=-1
):

    n, m = len(ref), len(seq)

    dp = np.zeros(
        (n + 1, m + 1)
    )

    for i in range(n + 1):
        dp[i][0] = i * gap

    for j in range(m + 1):
        dp[0][j] = j * gap

    for i in range(
        1,
        n + 1
    ):

        for j in range(
            1,
            m + 1
        ):

            s = (
                match
                if ref[i - 1] == seq[j - 1]
                else mismatch
            )

            dp[i][j] = max(
                dp[i - 1][j - 1] + s,
                dp[i - 1][j] + gap,
                dp[i][j - 1] + gap
            )

    i, j = n, m

    pairs = []

    while (
        i > 0
        or j > 0
    ):

        if (
            i > 0
            and j > 0
        ):

            s = (
                match
                if ref[i - 1] == seq[j - 1]
                else mismatch
            )

            if dp[i][j] == (
                dp[i - 1][j - 1] + s
            ):

                pairs.append(
                    (
                        i - 1,
                        seq[j - 1]
                    )
                )

                i -= 1
                j -= 1

                continue

        if (
            i > 0
            and dp[i][j]
            == dp[i - 1][j] + gap
        ):

            i -= 1

            continue

        j -= 1

    return list(
        reversed(pairs)
    )


def combine_readings(
    readings,
    credit=0.4
):

    readings = [
        r
        for r in readings
        if r["text"]
    ]

    if not readings:
        return "", []

    length_weight = {}

    for r in readings:

        L = len(
            r["text"]
        )

        length_weight[L] = (
            length_weight.get(
                L,
                0.0
            )
            +
            r["weight"]
            *
            r["conf"]
        )

    ref_len = max(
        length_weight,
        key=length_weight.get
    )

    same_len = [
        r
        for r in readings
        if len(r["text"]) == ref_len
    ]

    ref = max(
        same_len,
        key=lambda r:
        r["weight"] * r["conf"]
    )["text"]

    votes = [
        {}
        for _ in range(len(ref))
    ]

    for r in readings:

        for pos, ch in _align(
            ref,
            r["text"]
        ):

            if pos < len(votes):

                votes[pos][ch] = (
                    votes[pos].get(
                        ch,
                        0.0
                    )
                    +
                    r["weight"]
                    *
                    r["conf"]
                )

    for pos_votes in votes:

        if len(pos_votes) < 2:
            continue

        snapshot = dict(
            pos_votes
        )

        extra = {}

        for ch, w in snapshot.items():

            for alt in SIMILAR.get(
                ch,
                []
            ):

                if (
                    alt in snapshot
                    and alt != ch
                ):

                    extra[alt] = (
                        extra.get(
                            alt,
                            0.0
                        )
                        +
                        w * credit
                    )

        for alt, w in extra.items():

            pos_votes[alt] = (
                pos_votes.get(
                    alt,
                    0.0
                )
                + w
            )

    out_text = []
    out_conf = []

    for pos_votes in votes:

        if not pos_votes:

            out_text.append("?")
            out_conf.append(0.0)

            continue

        total = sum(
            pos_votes.values()
        )

        best_ch, best_w = max(
            pos_votes.items(),
            key=lambda kv: kv[1]
        )

        c = (
            best_w / total
            if total
            else 0.0
        )

        out_text.append(
            best_ch
            if c >= CHAR_THRESHOLD
            else "?"
        )

        out_conf.append(c)

    return (
        "".join(out_text),
        out_conf
    )


def reweight(
    readings,
    fused_text
):

    out = []

    for r in readings:

        n = min(
            len(r["text"]),
            len(fused_text)
        )

        agree = (
            sum(
                a == b
                for a, b in zip(
                    r["text"][:n],
                    fused_text[:n]
                )
            ) / n
            if n
            else 0.0
        )

        out.append(
            {
                **r,
                "weight":
                    r["weight"]
                    *
                    max(
                        0.2,
                        agree
                    )
            }
        )

    return out


def fuse(readings):

    first, _ = combine_readings(
        readings
    )

    if not first:
        return "", []

    return combine_readings(
        reweight(
            readings,
            first
        )
    )


# ============================================================
# INDIAN NUMBER PLATE FORMAT
# ============================================================

def repair_format(text):

    if len(text) == 10:

        template = "LLDDLLDDDD"

    elif len(text) == 9:

        template = "LLDDLDDDD"

    else:

        return text

    out = list(text)

    for i, want in enumerate(
        template
    ):

        ch = out[i]

        is_digit = ch.isdigit()

        if (
            want == "D"
            and not is_digit
        ):

            for alt in SIMILAR.get(
                ch,
                []
            ):

                if alt.isdigit():

                    out[i] = alt

                    break

        elif (
            want == "L"
            and is_digit
        ):

            for alt in SIMILAR.get(
                ch,
                []
            ):

                if alt.isalpha():

                    out[i] = alt

                    break

    return "".join(out)


def indian_format_bonus(text):

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


# ============================================================
# PER IMAGE OCR
# ============================================================

def read_plate(
    plate,
    rectified,
    quality
):

    sources = []

    for name, img in (
        ("raw", plate),
        ("rectified", rectified)
    ):

        if (
            img is None
            or img.size == 0
        ):
            continue

        for vname, variant in ocr_variants(
            img,
            quality
        ):

            sources.append(
                (
                    f"{name}-{vname}",
                    variant
                )
            )


    two_line = (
        split_two_line(
            rectified
        )
        if rectified is not None
        else None
    )


    if two_line is not None:

        top, bottom = two_line

        for half_name, half in (
            ("top", top),
            ("bottom", bottom)
        ):

            for vname, variant in ocr_variants(
                half,
                quality
            ):

                sources.append(
                    (
                        f"line-{half_name}-{vname}",
                        variant
                    )
                )


    readings = []

    for name, img in sources:

        for text, conf in _tesseract_read(
            img
        ):

            readings.append(
                {
                    "text": text,
                    "conf": conf,
                    "weight": 1.0,
                    "source":
                        f"tess-{name}"
                }
            )

        for text, conf in _rapidocr_read(
            img
        ):

            readings.append(
                {
                    "text": text,
                    "conf": conf,
                    "weight": 1.0,
                    "source":
                        f"rapid-{name}"
                }
            )


    if not readings:
        return "", [], []


    if two_line is not None:

        top_texts = [
            r["text"]
            for r in readings
            if "line-top"
            in r["source"]
        ]

        bottom_texts = [
            r["text"]
            for r in readings
            if "line-bottom"
            in r["source"]
        ]

        if (
            top_texts
            and bottom_texts
        ):

            combo = (
                max(
                    top_texts,
                    key=len
                )
                +
                max(
                    bottom_texts,
                    key=len
                )
            )

            readings.append(
                {
                    "text": combo,
                    "conf": 0.4,
                    "weight": 1.0,
                    "source":
                        "two-line-combo"
                }
            )


    text, char_conf = fuse(
        readings
    )

    return (
        text,
        char_conf,
        readings
    )


# ============================================================
# CONFIDENCE
# ============================================================

def confidence_score(
    ocr_conf,
    agreement,
    stability,
    quality,
    geometry
):

    weights = SCORE_WEIGHTS

    parts = {
        "ocr": ocr_conf,
        "agreement": agreement,
        "stability": stability,
        "quality": quality,
        "geometry": geometry
    }

    score = sum(
        weights.get(
            k,
            0.0
        ) * parts[k]
        for k in parts
    )

    return clamp(score)


def final_text(
    text,
    char_conf,
    score
):

    if (
        score < SCORE_THRESHOLD
        or not text
    ):

        return None, "unreadable"

    return (
        "".join(
            c
            if cc >= CHAR_THRESHOLD
            else "?"
            for c, cc
            in zip(
                text,
                char_conf
            )
        ),
        "ok"
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(image):

    plate, bbox, det_conf, backend = detect_plate(
        image
    )

    if plate is None:
        return None

    quality = check_quality(
        plate
    )

    rectified = rectify(
        plate
    )

    text, char_conf, readings = read_plate(
        plate,
        rectified,
        quality
    )

    ocr_conf = (
        float(
            np.mean(
                [
                    r["conf"]
                    for r in readings
                ]
            )
        )
        if readings
        else 0.0
    )

    return {
        "plate": plate,
        "rectified": rectified,
        "bbox": bbox,
        "det_conf": det_conf,
        "backend": backend,
        "quality": quality,
        "text": text,
        "char_conf": char_conf,
        "ocr_conf": ocr_conf,
        "readings": readings,
        "weight":
            max(
                quality["combined"],
                0.05
            )
    }


def process_group(images):

    results = [
        process_image(img)
        for img in images
    ]

    results = [
        r
        for r in results
        if r is not None
    ]

    if not results:
        return None

    per_image = [
        {
            "text": r["text"],
            "conf": r["ocr_conf"],
            "weight": r["weight"]
        }
        for r in results
    ]

    if len(results) > 1:

        fused_text, char_conf = fuse(
            per_image
        )

    else:

        fused_text = results[0]["text"]
        char_conf = results[0]["char_conf"]


    fused_text = repair_format(
        fused_text
    )


    agreement = (
        float(
            np.mean(
                [
                    (
                        sum(
                            a == b
                            for a, b
                            in zip(
                                r["text"][
                                    :len(fused_text)
                                ],
                                fused_text
                            )
                        )
                        /
                        len(fused_text)
                    )
                    if fused_text
                    else 0.0
                    for r in results
                ]
            )
        )
        if fused_text
        else 0.0
    )


    stability = (
        float(
            np.mean(char_conf)
        )
        if char_conf
        else 0.0
    )


    ocr_conf = float(
        np.mean(
            [
                r["ocr_conf"]
                for r in results
            ]
        )
    )


    quality = float(
        np.mean(
            [
                r["quality"]["combined"]
                for r in results
            ]
        )
    )


    geometry = float(
        np.mean(
            [
                r["quality"]["perspective"]
                for r in results
            ]
        )
    )


    score = confidence_score(
        ocr_conf,
        agreement,
        stability,
        quality,
        geometry
    )

    score = clamp(
        score +
        indian_format_bonus(
            fused_text
        )
    )


    text, status = final_text(
        fused_text,
        char_conf,
        score
    )


    return {
        "results": results,
        "fused_text": fused_text,
        "score": score,
        "text": text,
        "status": status,
        "agreement": agreement
    }


# ============================================================
# INPUT HELPERS
# ============================================================

def read_image_file(
    uploaded_file
):

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    return cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="License Plate Recognition",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)


st.markdown(
    """
<style>

/* =========================================================
   GLOBAL
========================================================= */

.stApp {
    background: #070c14;
    color: #eef3fb;
}

.block-container {
    max-width: 1180px;
    padding: 28px 28px 36px;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* =========================================================
   HEADER
========================================================= */

.app-title {
    font-size: 30px;
    line-height: 1.15;
    font-weight: 800;
    letter-spacing: -0.8px;
    margin: 0;
}

.app-subtitle {
    color: #8c9bb1;
    font-size: 13px;
    margin-top: 10px;
}

.ai-badge {
    border: 1px solid #304767;
    border-radius: 9px;
    padding: 10px 15px;
    min-width: 145px;
    background: #0b1320;
}

.ai-line {
    display: flex;
    align-items: center;
    gap: 9px;
    font-size: 12px;
    font-weight: 700;
}

.ai-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    background: #35d58c;
    box-shadow: 0 0 10px rgba(53,213,140,.35);
}

.ai-sub {
    color: #718199;
    font-size: 10px;
    margin-top: 5px;
}


/* =========================================================
   MAIN FRAME
========================================================= */

/*
IMPORTANT:
We use Streamlit's native border container instead of
opening/closing HTML divs around Streamlit widgets.

This prevents the empty rounded bar / overlap issue.
*/

[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #263b58 !important;
    border-radius: 13px !important;
    background: #0a111d !important;
    padding: 20px 17px 17px !important;
    margin-top: 32px !important;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    border: 0 !important;
}


/* =========================================================
   THREE STEPS
========================================================= */

.steps {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 28px;
    margin: 0 14px 24px;
}

.step {
    position: relative;
    min-height: 82px;
}

.step:after {
    content: "";
    position: absolute;
    left: 44px;
    right: -28px;
    top: 16px;
    height: 1px;
    background: #1d2e45;
}

.step:last-child:after {
    display: none;
}

.step-num {
    position: relative;
    z-index: 2;

    width: 32px;
    height: 32px;

    border-radius: 50%;

    display: grid;
    place-items: center;

    background: #edf3ff;
    color: #182337;

    font-weight: 800;
    font-size: 13px;
}

.step-title {
    font-size: 14px;
    font-weight: 750;
    margin-top: 11px;
}

.step-desc {
    color: #75859b;
    font-size: 11px;
    margin-top: 5px;
}


/* =========================================================
   WORKFLOW COLUMNS
========================================================= */

.panel-heading {
    font-size: 14px;
    font-weight: 750;
    margin-bottom: 7px;
}

.panel-copy {
    color: #e5eaf2;
    font-size: 12px;
    line-height: 1.65;
    min-height: 40px;
}


/* =========================================================
   FILE UPLOADER
========================================================= */

div[data-testid="stFileUploader"] {
    border: 1px dashed #3c5575;
    border-radius: 11px;
    background: #101a28;
    padding: 6px;
    margin-top: 7px;
}

div[data-testid="stFileUploaderDropzone"] {
    min-height: 132px;
}

div[data-testid="stFileUploaderDropzoneInstructions"] > div:first-child {
    color: #dce5f2;
}

div[data-testid="stFileUploaderDropzoneInstructions"] span,
div[data-testid="stFileUploaderDropzoneInstructions"] small {
    color: #8190a5;
}


/* =========================================================
   BUTTON
========================================================= */

.stButton > button {
    width: 100%;
    min-height: 42px;

    border: 0;
    border-radius: 8px;

    background: #315cf0;
    color: white;

    font-weight: 750;
    font-size: 13px;

    margin-top: 8px;
}

.stButton > button:hover {
    background: #3d68f6;
    color: white;
}

.stButton > button:disabled {
    background: #1d2b43;
    color: #74849a;
}


/* =========================================================
   IMAGE PREVIEW
========================================================= */

.image-card-title {
    font-size: 13px;
    font-weight: 750;
    margin: 17px 0 10px;
}

.image-shell {
    border: 1px solid #273b57;
    border-radius: 10px;
    background: #0c1522;
    min-height: 240px;
    overflow: hidden;
}

.image-placeholder {
    min-height: 240px;

    display: grid;
    place-items: center;

    color: #66778e;
    font-size: 12px;
}


/* =========================================================
   RESULT CARD
========================================================= */

.result-card {
    border: 1px solid #273b57;
    border-radius: 11px;

    background: #0c1522;

    margin-top: 28px;

    padding: 18px 18px 16px;
}

.result-top {
    display: flex;
    align-items: center;
    gap: 12px;
}

.result-icon {
    width: 31px;
    height: 31px;

    border-radius: 50%;

    display: grid;
    place-items: center;

    background: #243650;
    color: #b7c5d9;

    font-weight: 800;
}

.result-icon.success {
    background: #25cf83;
    color: #062117;
}

.result-title {
    font-size: 15px;
    font-weight: 800;
}

.result-note {
    color: #6f8097;
    font-size: 11px;
    margin-top: 4px;
}

.result-pill {
    margin-left: auto;

    border: 1px solid #25573f;
    background: #0b1c15;
    color: #5fe0a1;

    border-radius: 999px;

    padding: 5px 9px;

    font-size: 10px;
    font-weight: 700;
}


/* =========================================================
   PLATE NUMBER
========================================================= */

.plate-label {
    color: #6e8097;

    text-transform: uppercase;
    letter-spacing: 2px;

    font-size: 10px;

    text-align: center;

    margin-top: 22px;
}

.plate-text {
    color: #39df91;

    font-family: "Courier New", monospace;

    font-size: 31px;
    font-weight: 800;

    letter-spacing: 4px;

    text-align: center;

    margin: 7px 0 3px;

    word-break: break-all;
}

.plate-small {
    color: #64758c;

    font-size: 10px;

    text-align: center;
}


/* =========================================================
   CONFIDENCE BAR
========================================================= */

.progress {
    height: 6px;

    background: #1b2a3f;

    border-radius: 99px;

    overflow: hidden;

    margin: 15px auto 0;

    max-width: 480px;
}

.progress > div {
    height: 100%;

    background: #32d98e;

    border-radius: 99px;
}


/* =========================================================
   METRICS
========================================================= */

.metrics {
    display: grid;

    grid-template-columns: repeat(4, 1fr);

    gap: 8px;

    margin-top: 18px;
}

.metric {
    border: 1px solid #203149;

    border-radius: 8px;

    padding: 10px;

    background: #09121e;
}

.metric-label {
    color: #708097;

    font-size: 9px;

    text-transform: uppercase;

    letter-spacing: .7px;
}

.metric-value {
    color: #eef3fb;

    font-size: 16px;

    font-weight: 800;

    margin-top: 5px;
}

.metric-value.green {
    color: #4ce29c;
}

.metric-note {
    color: #586981;

    font-size: 9px;

    margin-top: 3px;
}


/* =========================================================
   DETAILS
========================================================= */

.details {
    display: grid;

    grid-template-columns: repeat(3, 1fr);

    gap: 12px;

    border-top: 1px solid #1b2a3e;

    margin-top: 15px;

    padding-top: 13px;
}

.detail-label {
    color: #64758b;
    font-size: 9px;
}

.detail-value {
    color: #c9d4e3;

    font-size: 11px;

    font-weight: 650;

    margin-top: 4px;
}


/* =========================================================
   EMPTY STATE
========================================================= */

.empty-card {
    border: 1px solid #273b57;

    border-radius: 11px;

    background: #0c1522;

    padding: 18px;

    margin-top: 28px;

    display: flex;
    align-items: center;

    gap: 12px;
}

.empty-icon {
    width: 32px;
    height: 32px;

    border-radius: 50%;

    background: #25364f;

    display: grid;
    place-items: center;

    color: #b4c0d2;
}

.empty-title {
    font-size: 13px;
    font-weight: 750;
}

.empty-text {
    color: #6e8098;
    font-size: 10px;
    margin-top: 4px;
}


/* =========================================================
   ERROR
========================================================= */

.error-card {
    border: 1px solid #5b3540;

    background: #1a0f14;

    color: #eaa5b0;

    border-radius: 9px;

    padding: 13px;

    margin-top: 15px;

    font-size: 11px;
}


/* =========================================================
   FOOTER
========================================================= */

.footer {
    border-top: 1px solid #1d2d43;

    margin-top: 31px;

    padding-top: 15px;

    display: flex;

    justify-content: space-between;

    color: #74849a;

    font-size: 9px;
}

.footer strong {
    color: #dce5f1;
}


/* =========================================================
   HIDE STREAMLIT TOOLBAR
========================================================= */

[data-testid="stToolbar"] {
    visibility: hidden;
}


/* =========================================================
   TABLET
========================================================= */

@media(max-width: 850px) {

    .steps {
        gap: 10px;
        margin-left: 5px;
        margin-right: 5px;
    }

    .step:after {
        right: -10px;
    }

    .metrics {
        grid-template-columns: repeat(2, 1fr);
    }

    .details {
        grid-template-columns: 1fr;
    }
}


/* =========================================================
   MOBILE
========================================================= */

@media(max-width: 650px) {

    .block-container {
        padding: 20px 14px 30px;
    }

    .steps {
        grid-template-columns: 1fr;
    }

    .step {
        min-height: 55px;
    }

    .step:after {
        display: none;
    }

    .app-title {
        font-size: 25px;
    }

    .ai-badge {
        display: none;
    }
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

head_l, head_r = st.columns(
    [5, 1],
    gap="medium"
)

with head_l:

    st.markdown(
        """
        <div class="app-title">
            License Plate Recognition
        </div>

        <div class="app-subtitle">
            Detect and read vehicle license plates using
            computer vision and OCR.
        </div>
        """,
        unsafe_allow_html=True
    )


with head_r:

    st.markdown(
        """
        <div class="ai-badge">

            <div class="ai-line">
                <span class="ai-dot"></span>
                AI Powered
            </div>

            <div class="ai-sub">
                YOLO + Tesseract
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN APPLICATION FRAME
# ============================================================

with st.container(border=True):


    # --------------------------------------------------------
    # THREE STEP HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="steps">

            <div class="step">

                <div class="step-num">
                    1
                </div>

                <div class="step-title">
                    Upload Image
                </div>

                <div class="step-desc">
                    Select a vehicle image
                </div>

            </div>


            <div class="step">

                <div class="step-num">
                    2
                </div>

                <div class="step-title">
                    Processing
                </div>

                <div class="step-desc">
                    Detect, clean and read the plate
                </div>

            </div>


            <div class="step">

                <div class="step-num">
                    3
                </div>

                <div class="step-title">
                    Results
                </div>

                <div class="step-desc">
                    Review the detected plate
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # WORKFLOW COLUMNS
    # --------------------------------------------------------

    col_upload, col_process, col_results = st.columns(
        [1.05, 1.05, 1.05],
        gap="medium"
    )


    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    with col_upload:

        st.markdown(
            """
            <div class="panel-heading">
                Upload Image
            </div>

            <div class="panel-copy"
                 style="color:#718198">
                Select one vehicle image
            </div>
            """,
            unsafe_allow_html=True
        )


        uploaded = st.file_uploader(
            "Drop Image Here - or - Click to Upload",

            type=[
                "jpg",
                "jpeg",
                "png",
                "webp",
                "avif"
            ],

            accept_multiple_files=False,

            label_visibility="collapsed",

            help="Only one image can be uploaded at a time."
        )


    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    with col_process:

        st.markdown(
            """
            <div class="panel-heading">
                Run Recognition
            </div>

            <div class="panel-copy">
                The model will detect the plate,
                clean the crop and run OCR.
            </div>
            """,
            unsafe_allow_html=True
        )


        run = st.button(
            "Recognize Plate",

            type="primary",

            use_container_width=True,

            disabled=uploaded is None
        )


    # --------------------------------------------------------
    # RESULTS INTRO
    # --------------------------------------------------------

    with col_results:

        st.markdown(
            """
            <div class="panel-heading">
                Results
            </div>

            <div class="panel-copy">
                Processed images and the detected
                plate number will appear below.
            </div>
            """,
            unsafe_allow_html=True
        )


    # ========================================================
    # SESSION STATE
    # ========================================================

    if "lpr_result" not in st.session_state:
        st.session_state.lpr_result = None

    if "lpr_source" not in st.session_state:
        st.session_state.lpr_source = None

    if "lpr_upload_name" not in st.session_state:
        st.session_state.lpr_upload_name = None


    # ========================================================
    # RESET WHEN IMAGE CHANGES
    # ========================================================

    if (
        uploaded is not None
        and uploaded.name
        != st.session_state.lpr_upload_name
    ):

        st.session_state.lpr_result = None

        st.session_state.lpr_source = (
            read_image_file(uploaded)
        )

        st.session_state.lpr_upload_name = (
            uploaded.name
        )


    elif uploaded is None:

        st.session_state.lpr_result = None

        st.session_state.lpr_source = None

        st.session_state.lpr_upload_name = None


    # ========================================================
    # RUN RECOGNITION
    # ========================================================

    if (
        uploaded is not None
        and run
    ):

        image = read_image_file(
            uploaded
        )

        with st.spinner(
            "Running recognition..."
        ):

            st.session_state.lpr_result = (
                process_group(
                    [image]
                )
            )

            st.session_state.lpr_source = image

            st.session_state.lpr_upload_name = (
                uploaded.name
            )


    # ========================================================
    # BUILD RESULT IMAGES
    # ========================================================

    result = st.session_state.lpr_result

    source_image = (
        st.session_state.lpr_source
    )

    annotated = None
    cleaned = None
    best = None


    if (
        result is not None
        and source_image is not None
    ):

        best = max(
            result["results"],
            key=lambda r:
            r["quality"]["combined"]
        )


        # ----------------------------------------------------
        # ANNOTATED IMAGE
        # ----------------------------------------------------

        annotated = source_image.copy()

        x1, y1, x2, y2 = best["bbox"]


        cv2.rectangle(
            annotated,

            (x1, y1),

            (x2, y2),

            (0, 230, 125),

            4
        )


        label = (
            result["text"]
            or result["fused_text"]
            or "PLATE DETECTED"
        )


        confidence_label = (
            f"{label}  "
            f"{result['score']:.0%}"
        )


        font = cv2.FONT_HERSHEY_SIMPLEX


        scale = max(
            0.55,
            min(
                1.15,
                source_image.shape[1] / 1200
            )
        )


        thickness = max(
            2,
            int(scale * 2.5)
        )


        (
            lw,
            lh
        ), baseline = cv2.getTextSize(
            confidence_label,
            font,
            scale,
            thickness
        )


        label_y = y1 - 10


        if (
            label_y
            - lh
            - baseline
            < 0
        ):

            label_y = (
                y2
                + lh
                + baseline
                + 10
            )


        box_top = (
            label_y
            - lh
            - baseline
            - 10
        )


        box_bottom = (
            label_y + 5
        )


        box_right = min(
            source_image.shape[1] - 1,
            x1 + lw + 16
        )


        cv2.rectangle(
            annotated,

            (x1, box_top),

            (box_right, box_bottom),

            (0, 185, 95),

            -1
        )


        cv2.putText(
            annotated,

            confidence_label,

            (
                x1 + 8,
                label_y - 4
            ),

            font,

            scale,

            (255, 255, 255),

            thickness,

            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # CLEANED PLATE
        # ----------------------------------------------------

        cleaned = (
            best["rectified"]
            if best["rectified"] is not None
            else best["plate"]
        )


    # ========================================================
    # IMAGE PREVIEW
    # ========================================================

    p1, p2, p3 = st.columns(
        3,
        gap="medium"
    )


    # --------------------------------------------------------
    # UPLOADED IMAGE
    # --------------------------------------------------------

    with p1:

        st.markdown(
            '<div class="image-card-title">'
            'Uploaded Image'
            '</div>',
            unsafe_allow_html=True
        )


        if source_image is not None:

            st.image(
                cv2.cvtColor(
                    source_image,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

        else:

            st.markdown(
                """
                <div class="image-shell">
                    <div class="image-placeholder">
                        No image uploaded
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    # --------------------------------------------------------
    # CLEANED PLATE
    # --------------------------------------------------------

    with p2:

        st.markdown(
            '<div class="image-card-title">'
            'Detected Plate (Cleaned)'
            '</div>',
            unsafe_allow_html=True
        )


        if cleaned is not None:

            if cleaned.ndim == 2:

                disp = cv2.cvtColor(
                    cleaned,
                    cv2.COLOR_GRAY2RGB
                )

            else:

                disp = cv2.cvtColor(
                    cleaned,
                    cv2.COLOR_BGR2RGB
                )


            st.image(
                disp,
                use_container_width=True
            )

        else:

            st.markdown(
                """
                <div class="image-shell">
                    <div class="image-placeholder">
                        Plate crop will appear here
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    # --------------------------------------------------------
    # ANNOTATED IMAGE
    # --------------------------------------------------------

    with p3:

        st.markdown(
            '<div class="image-card-title">'
            'Annotated Image'
            '</div>',
            unsafe_allow_html=True
        )


        if annotated is not None:

            st.image(
                cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                ),
                use_container_width=True
            )

        else:

            st.markdown(
                """
                <div class="image-shell">
                    <div class="image-placeholder">
                        Annotated result will appear here
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # EMPTY / RESULT STATE
    # ========================================================

    if result is None:

        st.markdown(
            """
            <div class="empty-card">

                <div class="empty-icon">
                    !
                </div>

                <div>

                    <div class="empty-title">
                        Waiting for an image
                    </div>

                    <div class="empty-text">
                        Upload a vehicle image and
                        click Recognize Plate.
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    else:

        if best is None:

            st.markdown(
                """
                <div class="error-card">
                    No license plate was detected.
                    Try a clearer or closer vehicle image.
                </div>
                """,
                unsafe_allow_html=True
            )


        else:

            # ------------------------------------------------
            # RESULT VALUES
            # ------------------------------------------------

            display_text = (
                result["text"]
                or result["fused_text"]
                or "UNREADABLE"
            )


            score_pct = (
                float(result["score"])
                * 100
            )


            if score_pct >= 85:

                confidence_word = "Very High"

            elif score_pct >= 70:

                confidence_word = "High"

            elif score_pct >= 50:

                confidence_word = "Moderate"

            else:

                confidence_word = "Low"


            # ------------------------------------------------
            # RESULT CARD
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-top">

                        <div class="result-icon success">
                            ✓
                        </div>

                        <div>

                            <div class="result-title">
                                Recognition Result
                            </div>

                            <div class="result-note">
                                Plate number and confidence scores
                            </div>

                        </div>

                        <div class="result-pill">
                            {confidence_word} confidence
                        </div>

                    </div>


                    <div class="plate-label">
                        Detected plate number
                    </div>


                    <div class="plate-text">
                        {display_text}
                    </div>


                    <div class="plate-small">
                        OCR reading · Indian vehicle plate
                    </div>


                    <div class="progress">

                        <div
                            style="
                                width:
                                {max(0, min(100, score_pct)):.1f}%;
                            "
                        ></div>

                    </div>


                    <div class="metrics">


                        <div class="metric">

                            <div class="metric-label">
                                Accuracy
                            </div>

                            <div class="metric-value green">
                                {score_pct:.1f}%
                            </div>

                            <div class="metric-note">
                                combined score
                            </div>

                        </div>


                        <div class="metric">

                            <div class="metric-label">
                                Detection
                            </div>

                            <div class="metric-value">
                                {best["det_conf"]:.1%}
                            </div>

                            <div class="metric-note">
                                plate localization
                            </div>

                        </div>


                        <div class="metric">

                            <div class="metric-label">
                                OCR Confidence
                            </div>

                            <div class="metric-value">
                                {best["ocr_conf"]:.1%}
                            </div>

                            <div class="metric-note">
                                text recognition
                            </div>

                        </div>


                        <div class="metric">

                            <div class="metric-label">
                                Image Quality
                            </div>

                            <div class="metric-value">
                                {best["quality"]["combined"]:.1%}
                            </div>

                            <div class="metric-note">
                                blur / exposure / noise
                            </div>

                        </div>


                    </div>


                    <div class="details">


                        <div>

                            <div class="detail-label">
                                Confidence strength
                            </div>

                            <div class="detail-value">
                                {confidence_word}
                            </div>

                        </div>


                        <div>

                            <div class="detail-label">
                                OCR agreement
                            </div>

                            <div class="detail-value">
                                {result["agreement"]:.1%}
                            </div>

                        </div>


                        <div>

                            <div class="detail-label">
                                Detection engine
                            </div>

                            <div class="detail-value">
                                {best["backend"].upper()}
                            </div>

                        </div>


                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # LOW OCR WARNING
            # ------------------------------------------------

            if not result["text"]:

                st.markdown(
                    """
                    <div class="error-card">

                        ⚠️ A plate was detected,
                        but the OCR reading did not pass
                        the reliability threshold.

                        The displayed text may contain
                        uncertain characters.

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <div>

            <strong>
                License Plate Recognition
            </strong>

            <br>

            Built with YOLO, OpenCV and Tesseract OCR

        </div>


        <div>

            <strong>
                Computer Vision Pipeline
            </strong>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)
