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
                               inner_ratio=0.45,
                               outer_r0=0.65,
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
    A = lab[..., 1].astype(np.float32)
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

        a_mean = float(np.median(A[whole]))
        feats.append((True, diff, a_mean))

    diffs = np.array([d for ok, d, m in feats if ok], dtype=np.float32)
    means = np.array([m for ok, d, m in feats if ok], dtype=np.float32)

    if diffs.size == 0:
        return ["unknown"] * len(circles)

    diff_th = float(np.median(diffs) + 2.0 * np.std(diffs))
    diff_th = max(diff_th, 3.0)  # minimum threshold for bimetal detection
    mean_th = float(np.median(means))

    materials = []
    for ok, diff, mean in feats:
        if not ok:
            materials.append("unknown")
        elif diff >= diff_th:
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
    diam_px = np.array([2.0 * float(r) for (_, _, r) in circles], dtype=np.float32)
    bimetal_idxs = [i for i, m in enumerate(materials) if m == "bimetal"]
    if len(bimetal_idxs) == 0:
        raise RuntimeError("No bimetal coin detected → cannot anchor scale")
    bi = max(bimetal_idxs, key=lambda i: diam_px[i])  # largest bimetal as anchor

    # Material correction (unchanged)
    d_mm_temp = diam_px / (diam_px[bi] / COINS_DIAM_MM[200])  # temp scale for correction
    materials = list(materials)
    for i in range(len(materials)):
        di = float(d_mm_temp[i])
        mi = materials[i]
        if mi == "copper" and di > 21.6:
            materials[i] = "gold"
        if mi == "gold" and di < 18.8:
            materials[i] = "copper"

    # Group indices (unchanged)
    idx_copper = [i for i, mat in enumerate(materials) if mat == "copper"]
    idx_gold = [i for i, mat in enumerate(materials) if mat == "gold"]
    idx_bimetal = [i for i, mat in enumerate(materials) if mat == "bimetal"]
    idx_unknown = [i for i, mat in enumerate(materials) if mat not in ("copper", "gold", "bimetal")]

    # Try both 1€ and 2€ for anchor, choose best by minimizing diameter error
    best_err = float('inf')
    best_d_mm = None
    best_scale = None
    best_values_list = None
    for candidate_type in [100, 200]:  # 1€ or 2€
        scale_try = float(diam_px[bi] / COINS_DIAM_MM[candidate_type])
        d_mm_try = diam_px / scale_try
        values_try = [None] * len(circles)

        # Assign by ranking (unchanged)
        assign_by_ranking(d_mm_try, values_try, idx_copper, MAT_GROUPS["copper"])
        assign_by_ranking(d_mm_try, values_try, idx_gold, MAT_GROUPS["gold"])
        for j in idx_bimetal:
            # For other bimetals, assign closest between 100/200
            dj = float(d_mm_try[j])
            v = 100 if abs(dj - COINS_DIAM_MM[100]) < abs(dj - COINS_DIAM_MM[200]) else 200
            values_try[j] = int(v)
        for j in idx_unknown:
            v = min(COINS_DIAM_MM.keys(), key=lambda k: abs(COINS_DIAM_MM[k] - float(d_mm_try[j])))
            values_try[j] = int(v)

        # Compute total error
        err_try = 0.0
        for k, v in enumerate(values_try):
            if v is not None:
                err_try += abs(float(d_mm_try[k]) - COINS_DIAM_MM[v])

        if err_try < best_err:
            best_err = err_try
            best_d_mm = d_mm_try
            best_scale = scale_try
            best_values_list = [int(v) for v in values_try]

    print(f"[ANCHOR] coin index={bi}, best_type={best_values_list[bi]}, scale(px/mm)={best_scale:.4f}, err={best_err:.2f}")
    return best_d_mm, best_values_list, best_scale

def enforce_material_constraints(material, cents_pred):
    """
    Enforce denomination constraints based on material group.
    Return (is_valid, corrected_cent_list).
    """
    allowed = MAT_GROUPS.get(material, MAT_GROUPS["unknown"])
    return (int(cents_pred) in allowed), allowed

def estimate_values_by_ratio(circles):
    if len(circles) == 0:
        return [], [], 0.0
    radii = [float(r) for (_, _, r) in circles]
    rmax = max(radii)
    if rmax <= 1e-6:
        return [], [], 0.0

    # Try different assumptions for largest coin
    candidate_largest = ["50c", "1€", "2€"]  # Possible max
    if len(circles) < 5:
        candidate_largest = ["50c", "20c", "1€"]
    best_err = float('inf')
    best_cents = []
    for largest_label in candidate_largest:
        ref_ratio_max = EURO_RATIOS[largest_label]
        scale = rmax / ref_ratio_max  # Pseudo-scale
        ratios_try = [r / scale for r in radii]  # Normalized ratios
        labels_try = []
        for ratio in ratios_try:
            best_label = min(EURO_RATIOS.keys(), key=lambda coin: abs(ratio - EURO_RATIOS[coin]))
            labels_try.append(best_label)
        # Compute error: sum of abs diff to refs
        err_try = sum(abs(ratio - EURO_RATIOS[label]) for ratio, label in zip(ratios_try, labels_try))
        if err_try < best_err:
            best_err = err_try
            best_labels = labels_try
    cents = [int(LABEL_TO_CENTS[label]) for label in best_labels]
    total = sum(COIN_VALUES_EUR[label] for label in best_labels)
    return best_labels, cents, total

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
    If bimetal-mm fails (no bimetal), use ratio method (method 1) as ultimate fallback.
    """
    try:
        d_mm, cents_mm, scale = estimate_values_by_bimetal_mm(img_bgr, circles, materials)
    except RuntimeError as e:
        print(f"[WARN] Bimetal scale failed: {e}. Falling back to ratio method.")
        _, cents_mm, _ = estimate_values_by_ratio(circles)
        d_mm = None
        scale = None

    # Load ref DB once (unchanged)
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
        mat = materials[i] if (materials is not None and i < len(materials)) else "unknown"
        print(f"[DBG] i={i} mat={mat} match_label={label} score={score} fallback={cents_mm[i]}")
        # define "match success" (unchanged)
        ok = (label is not None) and (label in LABEL_TO_CENTS) and (score is not None) and (score >= score_th)
        if ok:
            pred_cent = int(LABEL_TO_CENTS[label])
            valid, allowed = enforce_material_constraints(mat, pred_cent)
            if valid:
                cents_final.append(pred_cent)
                debug_used.append((i, "MATCH", label, score))
            else:
                cents_final.append(int(cents_mm[i]))
                debug_used.append((i, "FALLBACK_MAT", label, score, mat, allowed))
        else:
            cents_final.append(int(cents_mm[i]))
            debug_used.append((i, "FALLBACK", label, score))
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