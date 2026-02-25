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


from sklearn.cluster import KMeans  # Unsupervised ML cơ bản

def classify_material_adaptive(img_bgr, circles,
                               inner_ratio=0.50,
                               outer_r0=0.70,
                               outer_r1=0.95,
                               min_pixels=50,
                               s_var_th=2500.0,  # Tuned from sim/real (adjust per step 1)
                               h_copper_max=15.0):
    if circles is None or len(circles) == 0:
        return []

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    HH, WW = H.shape

    yy, xx = np.ogrid[:HH, :WW]

    materials = []
    for i, (cx, cy, r) in enumerate(circles):
        cx, cy, r = float(cx), float(cy), float(r)

        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        whole = dist < (outer_r1 * r)

        if whole.sum() < min_pixels:
            materials.append("unknown")
            print(f"[DBG_CLASSIFY] i={i} s_var=N/A (pixels ít) mat=unknown")
            continue

        s_var = np.var(S[whole])
        mean_h = np.mean(H[whole])
        mat = "bimetal" if s_var >= s_var_th else ("copper" if mean_h <= h_copper_max else "gold")
        materials.append(mat)

        print(f"[DBG_CLASSIFY] i={i} s_var={s_var:.1f} mean_h={mean_h:.1f} mat={mat}")  # Print chi tiết để tune

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
def estimate_values_by_px_ranking(circles, materials):
    """
    Fallback when no bimetal anchor is available.
    Use pixel diameter ranking inside each material group:
      - Sort coins by diameter (px)
      - Map smallest->smallest denom in allowed group, etc.
    Returns: (diam_px, cents_list)
    """
    if circles is None or len(circles) == 0:
        return np.array([], dtype=np.float32), []

    diam_px = np.array([2.0 * float(r) for (_, _, r) in circles], dtype=np.float32)
    values_list = [None] * len(circles)

    if materials is None or len(materials) != len(circles):
        materials = ["unknown"] * len(circles)

    # group indices by material
    idx_copper = [i for i, mat in enumerate(materials) if mat == "copper"]
    idx_gold = [i for i, mat in enumerate(materials) if mat == "gold"]
    idx_bimetal = [i for i, mat in enumerate(materials) if mat == "bimetal"]
    idx_unknown = [i for i, mat in enumerate(materials) if mat not in ("copper", "gold", "bimetal")]

    # NOTE: we cannot decide 1€ vs 2€ without anchor -> map bimetal by size ranking (100<200)
    # We'll still rank them: smaller->1€, larger->2€
    def _assign_rank_px(idxs, allowed_values):
        if not idxs:
            return
        idxs_sorted = sorted(idxs, key=lambda i: float(diam_px[i]))
        allowed_sorted = sorted(allowed_values, key=lambda v: COINS_DIAM_MM[v])  # order by real mm
        n = len(idxs_sorted)
        m = len(allowed_sorted)
        k = min(n, m)
        for j in range(k):
            values_list[idxs_sorted[j]] = int(allowed_sorted[j])
        if n > m:
            # extra coins -> nearest by absolute px (rough)
            for j in range(m, n):
                ii = idxs_sorted[j]
                # choose closest denom by relative position (best-effort)
                values_list[ii] = int(allowed_sorted[-1])

    _assign_rank_px(idx_copper, MAT_GROUPS["copper"])
    _assign_rank_px(idx_gold, MAT_GROUPS["gold"])

    # bimetal: rank -> 100 then 200 (if many, nearest by rank)
    if idx_bimetal:
        idxs_sorted = sorted(idx_bimetal, key=lambda i: float(diam_px[i]))
        # smaller bimetal -> 1€, larger -> 2€
        for j, ii in enumerate(idxs_sorted):
            values_list[ii] = 100 if j == 0 else 200

    # unknown: nearest by global ranking across all denoms
    for i in idx_unknown:
        # pick denom whose mm rank best matches px rank globally (simple nearest by normalized position)
        # fallback simpler: choose nearest by comparing to median group sizes is unreliable -> use 10c as neutral
        values_list[i] = 10

    # fill any None
    for i, v in enumerate(values_list):
        if v is None:
            values_list[i] = 10

    return diam_px, [int(v) for v in values_list]

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

COIN_DMM = np.array(list(COINS_DIAM_MM.values()), dtype=np.float32)

def _fit_residual(d_mm):
    # khoảng cách tới coin gần nhất (mm)
    return float(np.mean([np.min(np.abs(COIN_DMM - float(x))) for x in d_mm]))

def choose_anchor_scale(diam_px, anchor_idx):
    # thử cả 1€ và 2€
    candidates = []
    for typ in (100, 200):
        scale = float(diam_px[anchor_idx] / COINS_DIAM_MM[typ])
        d_mm = diam_px / scale
        res = _fit_residual(d_mm)
        candidates.append((res, typ, scale))
    candidates.sort(key=lambda t: t[0])
    return candidates[0]  # (best_res, best_type, best_scale)

def estimate_scale_from_all_coins(diam_px, s_min=1.0, s_max=60.0, steps=800):
    # grid search scale, chọn scale có residual nhỏ nhất
    best = None
    for s in np.linspace(s_min, s_max, steps):
        d_mm = diam_px / s
        # residual: median khoảng cách tới coin gần nhất
        errs = [min(abs(float(dm) - COINS_DIAM_MM[v]) for v in COINS_DIAM_MM) for dm in d_mm]
        res = float(np.median(errs))
        if best is None or res < best[0]:
            best = (res, float(s))
    return best  # (residual_mm, scale_px_per_mm)

def nearest_by_diameter(d_mm):
    return min(COINS_DIAM_MM.keys(), key=lambda v: abs(COINS_DIAM_MM[v] - float(d_mm)))

def choose_value_soft(d_mm, mat, lam=1.2):
    allowed = set(MAT_GROUPS.get(mat, MAT_GROUPS["unknown"]))
    best_v, best_s = None, 1e9
    for v in COINS_DIAM_MM:
        s = abs(float(d_mm) - COINS_DIAM_MM[v]) + (0.0 if v in allowed else lam)
        if s < best_s:
            best_s, best_v = s, v
    return int(best_v)

def scale_sanity(d_mm):
    ok = [(14.0 <= float(x) <= 28.0) for x in d_mm]
    return sum(ok) / max(len(ok), 1)

# if sanity < 0.7: refit scale using all coins

def estimate_values_by_bimetal_mm(img_bgr, circles, materials,
                                 sanity_th=0.70, anchor_res_th=0.90):
    """
    Robust baseline:
      - If bimetal exists: choose anchor scale by residual-fit (try 1€ vs 2€)
      - If anchor is bad OR no bimetal: fit scale from all coins
      - Assign denom by nearest diameter with soft material penalty
    Returns: d_mm, cents_list, scale(px/mm)
    """
    if circles is None or len(circles) == 0:
        return np.array([], dtype=np.float32), [], None

    diam_px = np.array([2.0 * float(r) for (_, _, r) in circles], dtype=np.float32)

    # normalize materials length
    if materials is None or len(materials) != len(circles):
        materials = ["unknown"] * len(circles)

    # ---- A) Try anchor by bimetal (if any) using residual-fit (NOT bimetal_type hard decision)
    scale = None
    d_mm = None
    used = "NONE"

    bimetal_idxs = [i for i, m in enumerate(materials) if m == "bimetal"]
    if len(bimetal_idxs) > 0:
        bi = max(bimetal_idxs, key=lambda i: float(diam_px[i]))
        best_res, coin_type, scale_try = choose_anchor_scale(diam_px, bi)
        d_mm_try = diam_px / scale_try
        sanity = scale_sanity(d_mm_try)

        # accept anchor only if it makes sense
        if (sanity >= sanity_th) and (best_res <= anchor_res_th):
            scale = float(scale_try)
            d_mm = d_mm_try
            used = f"ANCHOR(type={coin_type},res={best_res:.3f},sanity={sanity:.2f})"
        else:
            print(f"[ANCHOR_REJECT] idx={bi} type={coin_type} scale={scale_try:.4f} "
                  f"res={best_res:.3f} sanity={sanity:.2f} -> will refit all-coins")

    # ---- B) If no good anchor -> fit scale from all coins
    if scale is None:
        res_all, scale_all = estimate_scale_from_all_coins(diam_px)
        scale = float(scale_all)
        d_mm = diam_px / scale
        used = f"ALL_COINS(res={res_all:.3f})"

    print(f"[BASELINE_SCALE] {used} scale(px/mm)={scale:.4f}")

    # ---- C) Assign values using soft constraint by material + diameter
    cents_list = [choose_value_soft(d_mm[i], materials[i]) for i in range(len(circles))]

    return d_mm, [int(v) for v in cents_list], scale

def enforce_material_constraints(material, cents_pred):
    """
    Enforce denomination constraints based on material group.
    Return (is_valid, corrected_cent_list).
    """
    allowed = MAT_GROUPS.get(material, MAT_GROUPS["unknown"])
    return (int(cents_pred) in allowed), allowed

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
                                              score_th=8, area_tol=0.35):
    """
    Try ORB matching per coin. If matching is unreliable -> fallback to:
      - bimetal-mm baseline (if anchor exists)
      - otherwise px-ranking baseline (no anchor)
    Returns: d_mm_or_None, cents_final, scale_or_None
    """

    if circles is None or len(circles) == 0:
        return None, [], None

    # --- 1) Build a baseline that NEVER crashes ---
    d_mm = None
    scale = None
    cents_baseline = None

    try:
        d_mm, cents_baseline, scale = estimate_values_by_bimetal_mm(img_bgr, circles, materials)
        print(f"[BASELINE] using robust-mm (scale={scale:.4f})")
    except Exception as e:
        # should almost never happen now
        _, cents_px = estimate_values_by_px_ranking(circles, materials)
        d_mm, scale = None, None
        cents_baseline = cents_px
        print(f"[BASELINE] fallback px-ranking ({type(e).__name__}: {e})")

    # --- 2) Load ref DB once ---
    global _REF_DB
    if _REF_DB is None:
        _REF_DB = build_ref_db(REF_DIR, nfeatures=1200)

    cents_final = []

    for i, c in enumerate(circles):
        label, score = match_coin_orb_area(
            img_bgr, mask_bin_255, c, _REF_DB,
            area_tol=area_tol, nfeatures=900, use_ransac=True
        )

        mat = materials[i] if (materials is not None and i < len(materials)) else "unknown"

        ok = (label is not None) and (label in LABEL_TO_CENTS) and (score is not None) and (score >= score_th)

        if ok:
            pred_cent = int(LABEL_TO_CENTS[label])

            # material constraint gate
            valid, allowed = enforce_material_constraints(mat, pred_cent)
            if valid:
                cents_final.append(pred_cent)
                print(f"[DBG] i={i} mat={mat} MATCH {label} score={score}")
            else:
                cents_final.append(int(cents_baseline[i]))
                print(f"[DBG] i={i} mat={mat} MATCH_REJECT({pred_cent}) allowed={allowed} -> BASELINE={cents_baseline[i]}")
        else:
            cents_final.append(int(cents_baseline[i]))
            print(f"[DBG] i={i} mat={mat} NO_MATCH label={label} score={score} -> BASELINE={cents_baseline[i]}")

    return d_mm, cents_final, scale


def estimate_values(method_id, img_bgr, circles, materials, mask_bin_255=None):
    """
    0 = bimetal-mm (fallback px-ranking if no anchor)
    1 = ratio
    2 = CM06 matching (ORB+RANSAC) + fallback baseline
    """
    if method_id == 0:
        d_mm, cents, scale = estimate_values_by_bimetal_mm(img_bgr, circles, materials)
        return d_mm, cents, scale

    if method_id == 1:
        labels, cents, total = estimate_values_by_ratio(circles)
        return None, cents, None

    if method_id == 2:
        if mask_bin_255 is None:
            mask_bin_255 = _auto_mask_from_circles(img_bgr.shape[:2], circles)
        return estimate_values_by_matching_with_fallback(
            img_bgr, circles, materials, mask_bin_255
        )

    raise ValueError(f"Unknown CLASSIFY_METHOD_ID={method_id}. Use 0..2.")