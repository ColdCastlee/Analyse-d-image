import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable


# -----------------------------
# Ground-truth physical specs (mm)
# -----------------------------
COIN_SPECS = {
    1:   {"label": "1c",  "diam_mm": 16.25, "mat": "copper"},
    2:   {"label": "2c",  "diam_mm": 18.75, "mat": "copper"},
    5:   {"label": "5c",  "diam_mm": 21.25, "mat": "copper"},

    10:  {"label": "10c", "diam_mm": 19.75, "mat": "gold"},
    20:  {"label": "20c", "diam_mm": 22.25, "mat": "gold"},
    50:  {"label": "50c", "diam_mm": 24.25, "mat": "gold"},

    100: {"label": "1€",  "diam_mm": 23.25, "mat": "bimetal"},
    200: {"label": "2€",  "diam_mm": 25.75, "mat": "bimetal"},
}

MAT_ALLOWED = {
    "copper": [1, 2, 5],
    "gold":   [10, 20, 50],
    "bimetal":[100, 200],
    "unknown": list(COIN_SPECS.keys()),
}
COIN_DIAMS_MM = np.array([v["diam_mm"] for v in COIN_SPECS.values()], dtype=np.float32)
COIN_KEYS = list(COIN_SPECS.keys())

def _nearest_diam_err(dm: float) -> float:
    # distance to closest known coin diameter (mm)
    return float(np.min(np.abs(COIN_DIAMS_MM - float(dm))))

def estimate_scale_robust(diam_px: np.ndarray,
                          rel_tol: float = 0.03,
                          min_support: int = 3) -> Optional[float]:
    """
    Robust scale: generate many candidate scales s = diam_px[i] / diam_mm[k]
    then choose the scale cluster that minimizes median residual in mm.
    This does NOT rely on material classification.
    """
    if diam_px is None or len(diam_px) == 0:
        return None

    # Candidate scales
    cand = []
    for dpx in diam_px.tolist():
        for k in COIN_KEYS:
            cand.append(float(dpx) / float(COIN_SPECS[k]["diam_mm"]))
    if len(cand) < 10:
        return None

    s = np.sort(np.array(cand, dtype=np.float32))

    # Find densest scale cluster under relative tolerance
    best_i, best_j = 0, 0
    j = 0
    for i in range(len(s)):
        if j < i:
            j = i
        while j + 1 < len(s) and s[j + 1] <= s[i] * (1.0 + rel_tol):
            j += 1
        if (j - i) > (best_j - best_i):
            best_i, best_j = i, j

    support = best_j - best_i + 1
    if support < min_support:
        return None

    # Now score a few candidates around the cluster median by residual
    s0 = float(np.median(s[best_i:best_j + 1]))

    # local refine: keep inliers of s0 and evaluate by median residual
    inliers = s[(s >= s0 * (1.0 - rel_tol)) & (s <= s0 * (1.0 + rel_tol))]
    if inliers.size < min_support:
        return None

    # choose best by residual on actual coins
    best_scale = None
    best_res = 1e9
    for sc in np.percentile(inliers, [10, 30, 50, 70, 90]):
        d_mm = diam_px / float(sc)
        errs = np.array([_nearest_diam_err(x) for x in d_mm.tolist()], dtype=np.float32)
        res = float(np.median(errs))
        if res < best_res:
            best_res = res
            best_scale = float(sc)

    return best_scale

@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float  # radius in pixels


@dataclass
class CoinResult:
    cents: int
    label: str
    material: str
    diam_px: float
    diam_mm: Optional[float]
    scale_px_per_mm: Optional[float]
    score_orb_inliers: Optional[int]


# -----------------------------
# Preprocessing utilities
# -----------------------------
def _apply_clahe_on_l(img_bgr: np.ndarray,
                      clip_limit: float = 2.0,
                      tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L2 = clahe.apply(L)
    out = cv2.cvtColor(cv2.merge([L2, A, B]), cv2.COLOR_LAB2BGR)
    return out


def _grayworld_wb_simple(img_bgr: np.ndarray,
                         sat_max: int = 70,
                         min_pixels: int = 5000) -> np.ndarray:
    """
    Simple gray-world white balance:
      - estimate per-channel gain using only pixels with saturation <= sat_max.
      - avoids strong colorful regions dominating the WB estimate.
    Falls back to identity if too few eligible pixels.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1]
    mask = (S <= sat_max)
    if int(mask.sum()) < min_pixels:
        return img_bgr

    mean_bgr = img_bgr[mask].reshape(-1, 3).mean(axis=0).astype(np.float32)
    mean_gray = float(mean_bgr.mean())
    gains = mean_gray / (mean_bgr + 1e-6)

    out = img_bgr.astype(np.float32)
    out[..., 0] *= gains[0]
    out[..., 1] *= gains[1]
    out[..., 2] *= gains[2]
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """
    Photometric normalization:
      1) gray-world WB (simple)
      2) CLAHE on LAB-L
      3) mild denoise
    """
    wb = _grayworld_wb_simple(img_bgr)
    clahe = _apply_clahe_on_l(wb)
    den = cv2.bilateralFilter(clahe, d=5, sigmaColor=40, sigmaSpace=40)
    return den


def specular_mask_hsv(img_bgr: np.ndarray,
                      v_th: int = 235,
                      s_th: int = 45) -> np.ndarray:
    """
    Binary mask of likely specular highlight pixels (True = highlight).
    Heuristic: high V + low S.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1]
    V = hsv[..., 2]
    mask = (V >= v_th) & (S <= s_th)

    # Clean up tiny holes/spots
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=1)
    return mask_u8.astype(bool)


# -----------------------------
# Geometry: circle detection (optional)
# -----------------------------
def detect_circles_hough(img_bgr: np.ndarray,
                         dp: float = 1.2,
                         min_dist: Optional[float] = None,
                         param1: float = 140,
                         param2: float = 28,
                         min_radius: int = 20,
                         max_radius: int = 0) -> List[Circle]:
    """
    Convenience function. If you already have circles, skip it.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    if min_dist is None:
        min_dist = max(gray.shape) / 16.0

    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT,
        dp=dp, minDist=min_dist,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius
    )
    if circles is None:
        return []

    circles = np.round(circles[0]).astype(np.int32)
    return [Circle(float(x), float(y), float(r)) for (x, y, r) in circles]


# -----------------------------
# Mask helpers
# -----------------------------
def _radial_mask(dist2: np.ndarray, r: float, r0: float, r1: float) -> np.ndarray:
    r0_2 = (r0 * r) ** 2
    r1_2 = (r1 * r) ** 2
    return (dist2 >= r0_2) & (dist2 <= r1_2)


def _coin_dist2_grid(H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    yy, xx = np.ogrid[:H, :W]
    return xx.astype(np.float32), yy.astype(np.float32)


def _mad(x: np.ndarray) -> float:
    if x.size == 0:
        return 1e6
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-6)


# -----------------------------
# Material classification
# -----------------------------
def _ring_edge_score(L: np.ndarray, dist2: np.ndarray, r: float,
                     r0: float = 0.52, r1: float = 0.78,
                     min_pixels: int = 200) -> float:
    """
    Measure edge strength (median gradient magnitude) inside a radial band.
    Used to detect the metal boundary ring for bimetal coins.
    """
    # Gradient magnitude on L channel
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    band = _radial_mask(dist2, r, r0, r1)
    if int(band.sum()) < min_pixels:
        return 0.0
    return float(np.median(mag[band]))


def classify_materials(img_bgr: np.ndarray,
                       circles: List[Circle],
                       min_pixels: int = 200) -> Tuple[List[str], List[Dict]]:
    """
    Returns:
      materials: list in {"copper","gold","bimetal","unknown"}
      debug: per-coin diagnostics

    Improvements vs old version:
      - Photometric robustness: expects preprocess() already used outside, but still works.
      - Excludes specular highlights.
      - Adds ring-edge score to detect bimetal boundary even when color is tricky.
    """
    if not circles:
        return [], []

    H, W = img_bgr.shape[:2]
    xx, yy = _coin_dist2_grid(H, W)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV LAB uses 0..255 with 128 offset for a,b
    L = lab[..., 0]                  # 0..255
    A = lab[..., 1] - 128.0          # centered
    B = lab[..., 2] - 128.0          # centered

    spec = specular_mask_hsv(img_bgr)

    materials: List[str] = []
    debug: List[Dict] = []

    for c in circles:
        cx, cy, r = float(c.cx), float(c.cy), float(c.r)
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2

        # Radial regions
        inner = _radial_mask(dist2, r, 0.15, 0.45)
        outer = _radial_mask(dist2, r, 0.72, 0.95)
        mid   = _radial_mask(dist2, r, 0.55, 0.85)

        # Exclude specular highlights
        inner &= ~spec
        outer &= ~spec
        mid   &= ~spec

        if (int(mid.sum()) < min_pixels or
            int(inner.sum()) < min_pixels or
            int(outer.sum()) < min_pixels):
            materials.append("unknown")
            debug.append({"reason": "too_few_pixels", "mat": "unknown"})
            continue

        # Robust color stats
        a_in = float(np.median(A[inner])); b_in = float(np.median(B[inner]))
        a_out = float(np.median(A[outer])); b_out = float(np.median(B[outer]))
        a_mid = float(np.median(A[mid]));  b_mid = float(np.median(B[mid]))

        # Color separation for bimetal (robust)
        delta_ab = float(np.hypot(a_out - a_in, b_out - b_in))
        disp = (_mad(A[inner]) + _mad(B[inner]) + _mad(A[outer]) + _mad(B[outer]))
        bimetal_score = float(delta_ab / (disp + 1e-6))

        # Chroma sanity (avoid classifying gray-ish stuff as gold/copper)
        chrom_mid = float(np.hypot(a_mid, b_mid))

        # ---- NEW: edge-based boundary score for bimetal
        # ring boundary tends to have a strong edge near r~0.6-0.7
        edge_ring  = _ring_edge_score(L, dist2, r, 0.52, 0.78, min_pixels=min_pixels)
        edge_inner = _ring_edge_score(L, dist2, r, 0.20, 0.45, min_pixels=min_pixels)
        edge_outer = _ring_edge_score(L, dist2, r, 0.78, 0.95, min_pixels=min_pixels)
        edge_ratio = float(edge_ring / (0.5 * (edge_inner + edge_outer) + 1e-6))

        # Decision thresholds (tuned to be conservative)
        is_bimetal_color = (delta_ab >= 10.0) and (bimetal_score >= 0.9)
        is_bimetal_edge  = (edge_ratio >= 1.35) and (edge_ring >= 18.0)

        if is_bimetal_color or is_bimetal_edge:
            mat = "bimetal"
        else:
            if chrom_mid < 6.0:
                mat = "unknown"
            else:
                # gold tends to be "yellower" (b higher), copper "redder" (a higher)
                mat = "gold" if (b_mid >= a_mid + 3.0) else "copper"

        materials.append(mat)
        debug.append({
            "a_in": a_in, "b_in": b_in,
            "a_out": a_out, "b_out": b_out,
            "a_mid": a_mid, "b_mid": b_mid,
            "delta_ab": delta_ab,
            "bimetal_score": bimetal_score,
            "chrom_mid": chrom_mid,
            "edge_ring": edge_ring,
            "edge_inner": edge_inner,
            "edge_outer": edge_outer,
            "edge_ratio": edge_ratio,
            "is_bimetal_color": bool(is_bimetal_color),
            "is_bimetal_edge": bool(is_bimetal_edge),
            "mat": mat,
        })

    return materials, debug

def predict_bimetal_value(img_bgr: np.ndarray,
                          circle: Circle) -> Optional[int]:
    """
    Predict 1€ (100) vs 2€ (200) using ring orientation:
      - 1€: outer ring more yellow than inner
      - 2€: inner more yellow than outer
    Returns None if uncertain.
    """
    H, W = img_bgr.shape[:2]
    xx, yy = _coin_dist2_grid(H, W)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    B = lab[..., 2] - 128.0

    spec = specular_mask_hsv(img_bgr)

    cx, cy, r = circle.cx, circle.cy, circle.r
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2

    inner = _radial_mask(dist2, r, 0.15, 0.45) & (~spec)
    outer = _radial_mask(dist2, r, 0.72, 0.95) & (~spec)

    if int(inner.sum()) < 200 or int(outer.sum()) < 200:
        return None

    b_in = float(np.median(B[inner]))
    b_out = float(np.median(B[outer]))
    diff = b_out - b_in

    # If outer is distinctly more yellow -> 1€; if inner more yellow -> 2€
    if diff >= 4.0:
        return 100
    if diff <= -4.0:
        return 200
    return None


# -----------------------------
# Scale estimation (px per mm) via 1D voting
# -----------------------------
def estimate_scale_vote(diam_px: np.ndarray,
                        materials: List[str],
                        rel_tol: float = 0.025,
                        min_support: int = 2) -> Optional[float]:
    """
    Vote-based scale estimation:
      candidates: s_ij = diam_px[i] / diam_mm[j] for allowed j in material group.
      pick densest cluster in scale space under relative tolerance.
    """
    candidates: List[float] = []
    for i, dpx in enumerate(diam_px.tolist()):
        mat = materials[i] if i < len(materials) else "unknown"
        allowed = MAT_ALLOWED.get(mat, MAT_ALLOWED["unknown"])
        for v in allowed:
            candidates.append(float(dpx) / float(COIN_SPECS[v]["diam_mm"]))

    if len(candidates) < 2:
        return None

    s = np.sort(np.array(candidates, dtype=np.float32))
    best_i, best_j = 0, 0

    j = 0
    for i in range(len(s)):
        if j < i:
            j = i
        while j + 1 < len(s) and s[j + 1] <= s[i] * (1.0 + rel_tol):
            j += 1
        if (j - i) > (best_j - best_i):
            best_i, best_j = i, j

    support = best_j - best_i + 1
    if support < min_support:
        return None

    s0 = float(np.median(s[best_i:best_j + 1]))

    # refine: keep only candidates within tol of s0
    inliers = s[(s >= s0 * (1.0 - rel_tol)) & (s <= s0 * (1.0 + rel_tol))]
    if inliers.size < min_support:
        return None

    return float(np.median(inliers))


def estimate_scale_with_bimetal_anchor(diam_px: np.ndarray,
                                       circles: List[Circle],
                                       materials: List[str],
                                       img_bgr: np.ndarray) -> Optional[float]:
    """
    If bimetal exists, attempt to anchor scale using ring orientation.
    Else use robust fit (material-free).
    """
    bimetal_idxs = [i for i, m in enumerate(materials) if m == "bimetal"]

    # 1) Try bimetal anchors first
    anchor_scales: List[float] = []
    for i in bimetal_idxs:
        pred = predict_bimetal_value(img_bgr, circles[i])
        if pred in (100, 200):
            anchor_scales.append(float(diam_px[i]) / float(COIN_SPECS[pred]["diam_mm"]))

    if anchor_scales:
        return float(np.median(np.array(anchor_scales, dtype=np.float32)))

    # 2) Robust fit using all coins (no material dependency)
    return estimate_scale_robust(diam_px, rel_tol=0.03, min_support=3)


# -----------------------------
# Denomination assignment (geometry only)
# -----------------------------
def assign_by_diameter(diam_px: np.ndarray,
                       materials: List[str],
                       scale_px_per_mm: float,
                       lam: float = 0.8) -> Tuple[np.ndarray, List[int]]:
    """
    Soft material constraint:
      score = |dm - diam_mm[v]| + lam * I(v not in allowed(material))
    This prevents 1€ being forced into gold group when material is slightly wrong.
    """
    d_mm = diam_px / float(scale_px_per_mm)
    cents: List[int] = []

    for i, dm in enumerate(d_mm.tolist()):
        mat = materials[i] if i < len(materials) else "unknown"
        allowed = set(MAT_ALLOWED.get(mat, MAT_ALLOWED["unknown"]))

        best_v = None
        best_s = 1e9
        for v in COIN_KEYS:
            s = abs(float(dm) - float(COIN_SPECS[v]["diam_mm"])) + (0.0 if v in allowed else lam)
            if s < best_s:
                best_s, best_v = s, v

        cents.append(int(best_v))

    return d_mm.astype(np.float32), cents


# -----------------------------
# ORB reference matcher (optional tie-breaker)
# -----------------------------
@dataclass(frozen=True)
class RefEntry:
    label: str
    keypoints: Tuple  # tuple of cv2.KeyPoint
    desc: np.ndarray  # uint8 descriptors


class ORBRefDB:
    def __init__(self,
                 ref_dir: Path,
                 nfeatures: int = 1500):
        self.ref_dir = Path(ref_dir)
        self.orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=10)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.entries: List[RefEntry] = []
        self._build()

    @staticmethod
    def _prep_gray(img_bgr: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(g)
        return g

    def _build(self) -> None:
        # Expected structure:
        #   ref_dir/
        #     1c/*.jpg, *.png
        #     2c/*.jpg, *.png
        #     10c/*.jpg, *.png
        #     50c/*.jpg, *.png
        #     1e/*.jpg, *.png
        #     2e/*.jpg, *.png
        for sub in sorted(self.ref_dir.iterdir()):
            if not sub.is_dir():
                continue
            label = sub.name
            for p in sorted(sub.glob("*")):
                if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    continue
                img = cv2.imread(str(p))
                if img is None:
                    continue
                g = self._prep_gray(img)
                kp, desc = self.orb.detectAndCompute(g, None)
                if desc is None or len(kp) < 20:
                    continue
                self.entries.append(RefEntry(label=label, keypoints=tuple(kp), desc=desc))

        if not self.entries:
            raise RuntimeError(f"No usable reference images found in: {self.ref_dir}")

    def match(self,
              coin_bgr: np.ndarray,
              candidate_labels: Optional[Iterable[str]] = None,
              ratio: float = 0.75,
              min_good: int = 20,
              ransac_reproj: float = 3.5) -> Tuple[Optional[str], Optional[int]]:
        """
        Returns (best_label, inlier_count) or (None, None).
        """
        gq = self._prep_gray(coin_bgr)
        kpq, dq = self.orb.detectAndCompute(gq, None)
        if dq is None or len(kpq) < 20:
            return None, None

        allowed = set(candidate_labels) if candidate_labels is not None else None

        best_label = None
        best_inliers = 0

        for e in self.entries:
            if allowed is not None and e.label not in allowed:
                continue

            matches = self.bf.knnMatch(dq, e.desc, k=2)
            good = []
            for m_n in matches:
                if len(m_n) != 2:
                    continue
                m, n = m_n
                if m.distance < ratio * n.distance:
                    good.append(m)

            if len(good) < min_good:
                continue

            pts_q = np.float32([kpq[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            pts_t = np.float32([e.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(pts_t, pts_q, cv2.RANSAC, ransac_reproj)
            if mask is None:
                continue
            inliers = int(mask.ravel().sum())
            if inliers > best_inliers:
                best_inliers = inliers
                best_label = e.label

        if best_label is None:
            return None, None
        return best_label, best_inliers


# -----------------------------
# Main pipeline
# -----------------------------
def crop_coin(img_bgr: np.ndarray, circle: Circle, pad: float = 0.15) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    r = circle.r * (1.0 + pad)
    x0 = max(int(circle.cx - r), 0)
    y0 = max(int(circle.cy - r), 0)
    x1 = min(int(circle.cx + r), W - 1)
    y1 = min(int(circle.cy + r), H - 1)
    return img_bgr[y0:y1 + 1, x0:x1 + 1].copy()

def _refine_radius_by_radial_edge(gray_u8: np.ndarray,
                                  c: Circle,
                                  search_frac: float = 0.18,
                                  dr_step: float = 1.0,
                                  n_angles: int = 72,
                                  smooth_k: int = 7,
                                  ring_w: float = 2.0) -> Circle:
    """
    Refine circle radius by maximizing radial edge response around the boundary.

    - Sample gradient magnitude along multiple angles.
    - Search r' in [r*(1-search_frac), r*(1+search_frac)].
    - Pick r' with max median gradient response on a thin ring.

    Works well to separate 20c vs 50c when Hough radius is a bit off.
    """
    H, W = gray_u8.shape[:2]
    cx, cy, r0 = float(c.cx), float(c.cy), float(c.r)

    # Gradient magnitude
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # Candidate radii
    r_min = max(5.0, r0 * (1.0 - search_frac))
    r_max = min(min(H, W) * 0.9, r0 * (1.0 + search_frac))
    rs = np.arange(r_min, r_max + 1e-6, dr_step, dtype=np.float32)
    if rs.size < 3:
        return c

    # Angles
    ang = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False).astype(np.float32)
    cos_a = np.cos(ang); sin_a = np.sin(ang)

    best_r = r0
    best_score = -1.0

    # Evaluate each radius: sample mag on ring [r-w, r+w]
    for r in rs:
        w = max(1.0, ring_w)
        # sample 2 rings: r-w and r+w then average
        scores = []
        for rr in (r - w, r, r + w):
            xs = cx + rr * cos_a
            ys = cy + rr * sin_a
            xi = np.clip(xs, 0, W - 1).astype(np.int32)
            yi = np.clip(ys, 0, H - 1).astype(np.int32)
            scores.append(mag[yi, xi])

        s = (scores[0] + scores[1] + scores[2]) / 3.0  # (n_angles,)
        # robust score: median to ignore a few specular/outliers
        sc = float(np.median(s))
        if sc > best_score:
            best_score = sc
            best_r = float(r)

    # Small smoothing toward original (avoid over-jump)
    new_r = 0.65 * best_r + 0.35 * r0
    return Circle(cx=cx, cy=cy, r=float(new_r))


def refine_circles_radii(img_bgr: np.ndarray,
                         circles: List[Circle],
                         use_preprocess: bool = True) -> List[Circle]:
    """
    Refine radii for all circles using radial edge scoring.
    """
    if not circles:
        return []
    
    img = preprocess(img_bgr) if use_preprocess else img_bgr
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    refined = []
    for c in circles:
        refined.append(_refine_radius_by_radial_edge(gray, c))
    return refined


def classify_euro_coins(img_bgr: np.ndarray,
                        circles: Optional[List[Circle]] = None,
                        ref_db: Optional[ORBRefDB] = None,
                        orb_inlier_th: int = 35) -> List[CoinResult]:
    """
    Non-ML classifier for: 1c,2c,10c,50c,1€,2€.

    - If scale can be estimated: uses diameter-in-mm + material constraints.
    - If not: uses ORB (if provided) constrained by material group.
    """
    img = preprocess(img_bgr)

    if circles is None:
        circles = detect_circles_hough(img)

    if not circles:
        return []
    circles = refine_circles_radii(img_bgr, circles, use_preprocess=True)

    materials, _dbg = classify_materials(img, circles)

    diam_px = np.array([2.0 * c.r for c in circles], dtype=np.float32)

    scale = estimate_scale_with_bimetal_anchor(diam_px, circles, materials, img)

    d_mm = None
    cents_geom: Optional[List[int]] = None
    if scale is not None:
        d_mm, cents_geom = assign_by_diameter(diam_px, materials, scale)

    results: List[CoinResult] = []
    print("[SCALE]", scale, "diam_px=", diam_px.tolist(), "materials=", materials)

    for i, c in enumerate(circles):
        mat = materials[i] if i < len(materials) else "unknown"

        # Candidate labels based on material
        allowed_vals = MAT_ALLOWED.get(mat, MAT_ALLOWED["unknown"])
        allowed_labels = [
            ("1e" if v == 100 else "2e") if v >= 100 else COIN_SPECS[v]["label"]
            for v in allowed_vals
        ]
        # Note: for ORB ref folder naming, we use 1c/2c/10c/50c/1e/2e

        # Start from geometry if available
        geom_cent = None
        geom_label = None
        if cents_geom is not None:
            geom_cent = int(cents_geom[i])
            geom_label = COIN_SPECS[geom_cent]["label"]

        # ORB tie-breaker
        orb_label = None
        orb_inliers = None
        if ref_db is not None:
            coin_patch = crop_coin(img, c)
            # If scale is known, only try the 1 best geometric denom; else try within material group
            cand = None
            if geom_cent is not None:
                cand = [("1e" if geom_cent == 100 else "2e") if geom_cent >= 100 else COIN_SPECS[geom_cent]["label"]]
            else:
                cand = allowed_labels

            orb_label, orb_inliers = ref_db.match(coin_patch, candidate_labels=cand)

        # Decision rule:
        # - If ORB is strong, trust it (still implicitly constrained by candidates)
        # - Else fall back to geometry if we have scale
        final_cent = None
        final_label = None

        if orb_label is not None and orb_inliers is not None and orb_inliers >= orb_inlier_th:
            if orb_label in ("1e", "1€"):
                final_cent = 100
            elif orb_label in ("2e", "2€"):
                final_cent = 200
            elif orb_label.endswith("c"):
                final_cent = int(orb_label[:-1])
            else:
                # Unknown label format; ignore
                final_cent = geom_cent

        if final_cent is None:
            final_cent = geom_cent if geom_cent is not None else allowed_vals[0]

        final_label = COIN_SPECS[int(final_cent)]["label"]
        results.append(CoinResult(
            cents=int(final_cent),
            label=final_label,
            material=mat,
            diam_px=float(diam_px[i]),
            diam_mm=(float(d_mm[i]) if d_mm is not None else None),
            scale_px_per_mm=(float(scale) if scale is not None else None),
            score_orb_inliers=(int(orb_inliers) if orb_inliers is not None else None),
        ))

    return results
