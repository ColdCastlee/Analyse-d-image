import cv2
import numpy as np
import os
from core.matching import build_ref_db, match_coin_orb_area, LABEL_TO_CENTS
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))     # .../src/core
_SRC_DIR  = os.path.dirname(_THIS_DIR)                     # .../src
_ROOT_DIR = os.path.dirname(_SRC_DIR)                      # .../Analyse-d-image
REF_DIR   = os.path.join(_ROOT_DIR, "data", "ref")

COINS_DIAM_MM = {
    1: 16.25,
    2: 18.75,
    5: 21.25,
    10: 19.75,
    20: 22.25,
    50: 24.25,
    100: 23.25,
    200: 25.75
}

MAT_GROUPS = {
    "copper": [1, 2, 5],
    "gold": [10, 20, 50],
    "bimetal": [100, 200],
    "unknown": list(COINS_DIAM_MM.keys())
}

# fallou metric
EURO_RATIOS = {
    "2€": 1.00,
    "50c": 0.94,
    "1€": 0.90,
    "20c": 0.86,
    "5c": 0.82,
    "10c": 0.77,
    "2c": 0.73,
    "1c": 0.63
}
COIN_VALUES_EUR = {
    "2€": 2.00,
    "1€": 1.00,
    "50c": 0.50,
    "20c": 0.20,
    "10c": 0.10,
    "5c": 0.05,
    "2c": 0.02,
    "1c": 0.01
}
LABEL_TO_CENTS = {
    "2€": 200,
    "1€": 100,
    "50c": 50,
    "20c": 20,
    "10c": 10,
    "5c": 5,
    "2c": 2,
    "1c": 1
}


def classify_material_adaptive(img_bgr, circles,
                               inner_ratio=0.50,
                               outer_r0=0.70,
                               outer_r1=0.95,
                               min_pixels=50):
    """
    Return: list[str] in {"copper","gold","bimetal","unknown"} for each circle.

    Logic:
      - Work in LAB, use B channel.
      - Compute:
          diff = |median(B_outer) - median(B_inner)|
          mean = median(B_whole_coin)
      - Thresholds are adaptive from current image:
          diff_th = median(diffs) + 2*std(diffs)
          mean_th = median(means)
      - If diff > diff_th -> bimetal
        else -> gold if mean > mean_th else copper
    """

    if circles is None or len(circles) == 0:
        return []

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    B = lab[..., 2].astype(np.float32)
    H, W = B.shape

    # Pre-build grid once (faster + cleaner)
    yy, xx = np.ogrid[:H, :W]

    feats = []  # each item: (is_valid:bool, diff:float, mean:float)
    for (cx, cy, r) in circles:
        cx = float(cx); cy = float(cy); r = float(r)

        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        inner = dist < (inner_ratio * r)
        outer = (dist > (outer_r0 * r)) & (dist < (outer_r1 * r))
        whole = dist < (outer_r1 * r)

        if inner.sum() < min_pixels or outer.sum() < min_pixels or whole.sum() < min_pixels:
            feats.append((False, 0.0, 0.0))
            continue

        # median is often more robust than mean (less sensitive to highlights)
        b_inner = float(np.median(B[inner]))
        b_outer = float(np.median(B[outer]))
        diff = float(abs(b_outer - b_inner))

        b_mean = float(np.median(B[whole]))
        feats.append((True, diff, b_mean))

    diffs = np.array([d for ok, d, m in feats if ok], dtype=np.float32)
    means = np.array([m for ok, d, m in feats if ok], dtype=np.float32)

    if diffs.size == 0:
        return ["unknown"] * len(circles)

    diff_th = float(np.median(diffs) + 2.0 * np.std(diffs))
    mean_th = float(np.median(means))

    materials = []
    for ok, diff, mean in feats:
        if not ok:
            materials.append("unknown")
        elif diff > diff_th:
            materials.append("bimetal")
        else:
            materials.append("gold" if mean > mean_th else "copper")

    return materials


def bimetal_type_1e_or_2e(img_bgr, cx, cy, r_px):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    B = lab[..., 2].astype(np.float32)

    H, W = B.shape
    cx, cy = int(cx), int(cy)
    r = float(r_px)

    yy, xx = np.ogrid[:H, :W]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    outer = (dist >= 0.72 * r) & (dist <= 0.95 * r)
    inner = (dist >= 0.10 * r) & (dist <= 0.45 * r)

    if outer.sum() < 50 or inner.sum() < 50:
        return 100

    b_outer = float(np.median(B[outer]))
    b_inner = float(np.median(B[inner]))
    return 100 if (b_outer - b_inner) > 3 else 200


def assign_by_ranking(d_mm, values_list, idxs, allowed_values):
    if len(idxs) == 0:
        return

    idxs_sorted = sorted(idxs, key=lambda i: float(d_mm[i]))
    allowed_sorted = sorted(allowed_values, key=lambda v: COINS_DIAM_MM[v])

    n = len(idxs_sorted)
    m = len(allowed_sorted)
    k = min(n, m)

    for i in range(k):
        values_list[idxs_sorted[i]] = int(allowed_sorted[i])

    if n > m:
        for j in range(m, n):
            ii = idxs_sorted[j]
            values_list[ii] = int(min(allowed_sorted, key=lambda v: abs(COINS_DIAM_MM[v] - float(d_mm[ii]))))


def estimate_values_by_bimetal_mm(img_bgr, circles, materials):
    """
    Your original: anchor scale with bimetal -> mm -> match by groups.
    Returns: d_mm, values_list(cents), scale(px/mm)
    """
    diam_px = np.array([2.0 * float(r) for (_, _, r) in circles], dtype=np.float32)

    bimetal_idxs = [i for i, m in enumerate(materials) if m == "bimetal"]
    if len(bimetal_idxs) == 0:
        raise RuntimeError("No bimetal coin detected → cannot anchor scale")

    bi = max(bimetal_idxs, key=lambda i: diam_px[i])
    cx, cy, r = circles[bi]
    coin_type = bimetal_type_1e_or_2e(img_bgr, cx, cy, r)

    scale = float(diam_px[bi] / COINS_DIAM_MM[coin_type])  # px/mm
    print(f"[ANCHOR] coin index={bi}, type={coin_type}, scale(px/mm)={scale:.4f}")

    d_mm = diam_px / scale
    values_list = [None] * len(circles)

    idx_copper = [i for i, mat in enumerate(materials) if mat == "copper"]
    idx_gold = [i for i, mat in enumerate(materials) if mat == "gold"]
    idx_bimetal = [i for i, mat in enumerate(materials) if mat == "bimetal"]
    idx_unknown = [i for i, mat in enumerate(materials) if mat not in ("copper", "gold", "bimetal")]

    assign_by_ranking(d_mm, values_list, idx_copper, MAT_GROUPS["copper"])
    assign_by_ranking(d_mm, values_list, idx_gold, MAT_GROUPS["gold"])

    for i in idx_bimetal:
        cx, cy, r = circles[i]
        values_list[i] = int(bimetal_type_1e_or_2e(img_bgr, cx, cy, r))

    for i in idx_unknown:
        v = min(COINS_DIAM_MM.keys(), key=lambda k: abs(COINS_DIAM_MM[k] - float(d_mm[i])))
        values_list[i] = int(v)

    values_list = [int(v) for v in values_list]
    return d_mm, values_list, scale


def estimate_values_by_ratio(circles):
    """
    fallou method
    """
    if len(circles) == 0:
        return [], [], 0.0

    radii = [float(r) for (_, _, r) in circles]
    rmax = max(radii)
    if rmax <= 1e-6:
        return [], [], 0.0

    labels = []
    cents = []
    total = 0.0

    for (_, _, r) in circles:
        ratio = float(r) / rmax

        best_label = None
        best_diff = 1e9
        for coin, ref in EURO_RATIOS.items():
            diff = abs(ratio - ref)
            if diff < best_diff:
                best_diff = diff
                best_label = coin

        labels.append(best_label)
        cents.append(int(LABEL_TO_CENTS[best_label]))
        total += float(COIN_VALUES_EUR[best_label])

    return labels, cents, total

from core.matching import build_ref_db, match_coin_orb_area, LABEL_TO_CENTS

_REF_DB = None

def _auto_mask_from_circles(shape_hw, circles):
    """
    Build a binary mask (255 inside circles) if mask_bin_255 is not provided.
    shape_hw: (H, W) of image
    """
    H, W = shape_hw
    m = np.zeros((H, W), dtype=np.uint8)
    for (cx, cy, r) in circles:
        cv2.circle(m, (int(cx), int(cy)), int(round(r)), 255, -1)
    return m

def estimate_values_by_matching_with_fallback(img_bgr, circles, materials, mask_bin_255,
                                              score_th=18, area_tol=0.35):
    """
    Try matching each detected coin against REF_DIR (front-side refs).
    If matching is unreliable -> fallback to bimetal-mm (method 0) result.
    Returns: d_mm (from mm method), cents_final, scale (from mm method)
    """

    # 1) Fallback baseline (bimetal-mm)
    d_mm, cents_mm, scale = estimate_values_by_bimetal_mm(img_bgr, circles, materials)

    # 2) Load ref DB once
    global _REF_DB
    if _REF_DB is None:
        _REF_DB = build_ref_db(REF_DIR, nfeatures=900)

    cents_final = []
    debug_used = []  # optional: store info

    for i, c in enumerate(circles):
        label, score = match_coin_orb_area(
            img_bgr, mask_bin_255, c, _REF_DB,
            area_tol=area_tol, nfeatures=900, use_ransac=True
        )

        # define "match success"
        ok = (label is not None) and (label in LABEL_TO_CENTS) and (score is not None) and (score >= score_th)

        if ok:
            cents_final.append(int(LABEL_TO_CENTS[label]))
            debug_used.append((i, "MATCH", label, score))
        else:
            # fallback to mm-based classification
            cents_final.append(int(cents_mm[i]))
            debug_used.append((i, "MM_FALLBACK", label, score))

    return d_mm, cents_final, scale


def estimate_values(method_id, img_bgr, circles, materials, mask_bin_255=None):
    """
    0 = bimetal-mm
    1 = ratio
    2 = CM06 matching (ORB+RANSAC) + area-scale (uses REF_DIR inside this file)
    """
    if method_id == 0:
        d_mm, cents, scale = estimate_values_by_bimetal_mm(img_bgr, circles, materials)
        return d_mm, cents, scale

    if method_id == 1:
        labels, cents, total = estimate_values_by_ratio(circles)
        return None, cents, None

    if method_id == 2:
    # if user didn't pass mask, build a coarse one from circles
        if mask_bin_255 is None:
            mask_bin_255 = _auto_mask_from_circles(img_bgr.shape[:2], circles)

        return estimate_values_by_matching_with_fallback(
            img_bgr, circles, materials, mask_bin_255
        )

    raise ValueError(f"Unknown CLASSIFY_METHOD_ID={method_id}. Use 0..2.")