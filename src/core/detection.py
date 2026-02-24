import cv2
import numpy as np
def _dedup_circles(circles, center_frac=0.35, r_frac=0.25):
    """
    Remove duplicate circle detections (multiple circles for the same coin).

    Parameters
    ----------
    circles : list of (cx, cy, r)
        Detected circles (center x, center y, radius).

    center_frac : float
        Maximum allowed center distance (relative to radius)
        to consider two circles as duplicates.

    r_frac : float
        Maximum allowed radius difference (relative to radius)
        to consider two circles as duplicates.

    Returns
    -------
    list of (cx, cy, r)
        Filtered list of circles with duplicates removed.
    """

    if not circles:
        return []

    # Sort by radius descending (keep larger circle first)
    circles = sorted(circles, key=lambda t: -t[2])

    kept = []
    for (cx, cy, r) in circles:
        is_duplicate = False

        for (kx, ky, kr) in kept:
            center_dist = np.hypot(cx - kx, cy - ky)

            # If centers are close and radii are similar → same coin
            if (
                center_dist < center_frac * min(r, kr) and
                abs(r - kr) < r_frac * min(r, kr)
            ):
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append((cx, cy, r))

    return kept

def detect_circles_cc_hough(img_bgr, enhanced, mask):
    circles = []

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas_all = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    if len(areas_all) == 0:
        return []

    med_area = float(np.median(areas_all))
    min_area = 0.50 * med_area
    split_area_th = 1.8 * med_area

    print("CC count:", num - 1, "median area:", med_area, "min_area:", min_area, "split_th:", split_area_th)

    for i in range(1, num):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        pad = 25
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(img_bgr.shape[1], x + w + pad), min(img_bgr.shape[0], y + h + pad)

        roi_gray = enhanced[y0:y1, x0:x1].copy()
        roi_mask = mask[y0:y1, x0:x1]

        bg_val = int(np.median(roi_gray))
        roi_gray[roi_mask == 0] = bg_val

        if area > split_area_th:
            roi_blur = cv2.medianBlur(roi_gray, 5)

            r_guess = np.sqrt((area / 2.0) / np.pi)
            minR = int(max(10, 0.65 * r_guess))
            maxR = int(1.35 * r_guess)
            minDist = int(max(20, 1.2 * r_guess))

            c = cv2.HoughCircles(
                roi_blur, cv2.HOUGH_GRADIENT,
                dp=1.2, minDist=minDist,
                param1=120, param2=22,
                minRadius=minR, maxRadius=maxR
            )

            if c is not None:
                c = np.squeeze(c).astype(np.float32)
                if c.ndim == 1:
                    c = c[None, :]

                kept = []
                for (cx, cy, r) in c:
                    cx_i, cy_i = int(round(cx)), int(round(cy))
                    if 0 <= cx_i < roi_mask.shape[1] and 0 <= cy_i < roi_mask.shape[0]:
                        if roi_mask[cy_i, cx_i] > 0:
                            kept.append((cx, cy, r))

                # Keep up to 2 circles from Hough candidates
                kept = sorted(kept, key=lambda t: -t[2])[:2]

                if len(kept) >= 1:
                    # Always accept the best one
                    cx1, cy1, r1 = kept[0]
                    circles.append((int(x0 + cx1), int(y0 + cy1), float(r1)))

                    # Accept a second one ONLY if clearly separated (likely two coins)
                    if len(kept) == 2:
                        cx2, cy2, r2 = kept[1]
                        dist = float(np.hypot(cx1 - cx2, cy1 - cy2))

                        # If too close -> shadow / duplicate
                        if dist > 1.45 * min(r1, r2):
                            circles.append((int(x0 + cx2), int(y0 + cy2), float(r2)))

                    continue
            # --- final dedup (NMS-like) ---
            circles_sorted = sorted(circles, key=lambda t: -t[2])  # larger first
            kept_final = []
            for (cx, cy, r) in circles_sorted:
                ok = True
                for (kx, ky, kr) in kept_final:
                    dist = float(np.hypot(cx - kx, cy - ky))
                    # if centers are too close and radii similar -> duplicate
                    if dist < 0.60 * min(r, kr) and abs(r - kr) / max(kr, 1e-6) < 0.35:
                        ok = False
                        break
                if ok:
                    kept_final.append((cx, cy, r))
            circles = kept_final

            cx, cy = centroids[i]
            r = float(np.sqrt(area / np.pi))
            circles.append((int(cx), int(cy), r))
            continue

        cx, cy = centroids[i]
        r = float(np.sqrt(area / np.pi))
        circles.append((int(cx), int(cy), r))

    # Remove duplicate detections (same coin detected multiple times)
    before = len(circles)
    circles = _dedup_circles(circles)
    after = len(circles)

    print(f"Detected coins (raw): {before}, after dedup: {after}")
    return circles

def detect_circles_contours_min_enclosing(mask, min_radius_px=60):
    """
    contours -> minEnclosingCircle -> filter by min radius
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("Aucune pièce détectée")
        exit()
    
    circles = []
    for ctr in contours:
        (x, y), radius = cv2.minEnclosingCircle(ctr)
        if radius > float(min_radius_px):
            circles.append((int(round(x)), int(round(y)), float(radius)))

    print("Contours count:", len(contours), "Detected circles:", len(circles), f"(minR={min_radius_px})")
    return circles

import cv2
import numpy as np


def detect_circles_hough_pure(enhanced,
                              dp=1.2,
                              min_dist_frac=0.08,
                              param1=120,
                              param2=28,
                              min_r_frac=0.04,
                              max_r_frac=0.20,
                              dedup=True):
    """
    PURE HOUGH on preprocessed image (enhanced = gray+blur+CLAHE).
    No mask, no CC, no DT.
    """

    if enhanced is None:
        return []

    h, w = enhanced.shape[:2]
    min_dim = min(h, w)

    minRadius = max(8, int(min_dim * min_r_frac))
    maxRadius = max(minRadius + 2, int(min_dim * max_r_frac))
    minDist = max(12, int(min_dim * min_dist_frac))

    circles = cv2.HoughCircles(
        enhanced,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    if circles is None:
        return []

    circles = np.round(circles[0]).astype(int)
    circles = [(int(x), int(y), int(r)) for (x, y, r) in circles]

    if not dedup:
        return circles

    # ---- concentric + dedup nhẹ (Hough hay bắt inner/outer) ----
    circles = sorted(circles, key=lambda t: -t[2])  # giữ vòng lớn trước
    kept = []
    for (x, y, r) in circles:
        dup = False
        for (kx, ky, kr) in kept:
            if np.hypot(x - kx, y - ky) < 0.20 * min(r, kr):
                dup = True
                break
        if not dup:
            kept.append((x, y, r))

    return kept
def detect_circles_hough_dynamic(
    enhanced,
    dp=1.2,
    param1=180,
    warm_param2=25,
    warm_min_r_frac=0.020,
    warm_max_r_frac=0.20,
    warm_min_dist_frac=0.08,
    sweep_param2=(30, 35, 40, 45, 50, 55, 60, 65),
    dedup=True,
    debug=False
):
    """
    Dynamic PURE HOUGH (per-image auto scale):
      1) Warmup Hough with low threshold to collect candidates.
      2) Estimate radius band from candidate radii (percentiles).
      3) Set minDist based on median radius.
      4) Sweep param2 and pick a reasonable/stable result.

    Input:
      enhanced: grayscale preprocessed image (gray+blur+CLAHE)
    Output:
      list of circles (cx, cy, r) in pixels
    """
    if enhanced is None:
        return []

    h, w = enhanced.shape[:2]
    min_dim = min(h, w)

    # -------------------------
    # 1) Warmup Hough: permissive
    # -------------------------
    warm_minR = max(8, int(min_dim * warm_min_r_frac))
    warm_maxR = max(warm_minR + 2, int(min_dim * warm_max_r_frac))
    warm_minDist = max(12, int(min_dim * warm_min_dist_frac))

    warm = cv2.HoughCircles(
        enhanced, cv2.HOUGH_GRADIENT,
        dp=dp, minDist=warm_minDist,
        param1=param1, param2=warm_param2,
        minRadius=warm_minR, maxRadius=warm_maxR
    )

    if warm is None:
        return []

    cand = np.round(warm[0]).astype(int)
    if cand.ndim != 2 or cand.shape[1] < 3:
        return []

    # --- radii from warm candidates ---
    radii = cand[:, 2].astype(np.float32)
    radii = radii[radii > 0]

    # ---- filter radii by density ----
    if len(radii) >= 4:
        hist, bin_edges = np.histogram(radii, bins=12)
        best_bin = int(np.argmax(hist))
        r_lo = float(bin_edges[best_bin])
        r_hi = float(bin_edges[best_bin + 1])
        radii = radii[(radii >= r_lo * 0.85) & (radii <= r_hi * 1.15)]

    # ---- fallback: if too few candidates after filtering ----
    circles_warm = [(int(x), int(y), int(r)) for (x, y, r) in cand]
    if dedup:
        circles_warm = _dedup_circles(circles_warm, center_frac=0.30, r_frac=0.22)

    if len(radii) < 4:
        if debug:
            print("[DYN] few warm candidates -> return warm:", len(circles_warm))
        return circles_warm

    # -------------------------
    # 2) Estimate radius band robustly
    #    (avoid tiny texture circles and huge false rings)
    # -------------------------
    r20 = float(np.percentile(radii, 20))
    r50 = float(np.percentile(radii, 50))
    r90 = float(np.percentile(radii, 90))

    # Expand slightly, clamp
    minR = max(8, int(0.80 * r20))
    maxR = max(minR + 2, int(1.15 * r90))

    # safety clamps vs image size
    maxR = min(maxR, int(0.30 * min_dim))

    # -------------------------
    # 3) minDist based on typical radius
    # -------------------------
    minDist = int(max(14, 1.35 * r50))

    if debug:
        print(f"[DYN] warm_n={len(cand)} r20={r20:.1f} r50={r50:.1f} r90={r90:.1f} -> minR={minR} maxR={maxR} minDist={minDist}")

    # -------------------------
    # 4) Sweep param2 and choose result (scoring)
    # -------------------------
    def score(cs):
        if len(cs) == 0:
            return -1e9
        rs = np.array([c[2] for c in cs], np.float32)
        return float(len(rs) - 0.8 * np.std(rs))  # many + consistent radii

    best = []
    best_p2 = None
    best_score = -1e9
    prev_n = None

    for p2 in sweep_param2:
        c = cv2.HoughCircles(
            enhanced, cv2.HOUGH_GRADIENT,
            dp=dp, minDist=minDist,
            param1=param1, param2=int(p2),
            minRadius=minR, maxRadius=maxR
        )
        if c is None:
            continue

        circles = np.round(c[0]).astype(int)
        circles = [(int(x), int(y), int(r)) for (x, y, r) in circles]
        if dedup:
            circles = _dedup_circles(circles, center_frac=0.30, r_frac=0.22)

        n = len(circles)
        if debug:
            print(f"[DYN] p2={p2:>2} -> n={n}")

        # Explosion guard (optional but helpful)
        if prev_n is not None and n > prev_n * 1.8 and (n - prev_n) >= 8:
            prev_n = n
            continue
        prev_n = n

        s = score(circles)
        if s > best_score:
            best_score = s
            best = circles
            best_p2 = p2

    if debug:
        print(f"[DYN] chosen p2={best_p2} score={best_score:.2f} n={len(best)}")

    return best
def detect_circles(method_id, img_bgr, enhanced, mask):
    """
    Returns circles as list of (cx, cy, r_px)
    """
    if method_id == 0:
        return detect_circles_cc_hough(img_bgr, enhanced, mask)

    if method_id == 1:
        return detect_circles_contours_min_enclosing(mask, min_radius_px=60)

    if method_id == 2:
        return detect_circles_hough_pure(
            enhanced,
            dp=1.1,              # coarser accumulator -> bớt nhạy
            min_dist_frac=0.15,  # tăng minDist để tránh nhiều vòng gần nhau
            param1=200,          # edge mạnh hơn -> ít nhiễu nền
            param2=60,           # QUAN TRỌNG: tăng nữa để giảm circle ảo
            min_r_frac=0.035,    # bỏ vòng nhỏ (nền/texture)
            max_r_frac=0.13,     # bỏ vòng quá to (hay là vòng ảo)
            dedup=True
        )
    if method_id == 3:
        # Dynamic PURE HOUGH (auto scale per image)
        return detect_circles_hough_dynamic(
            enhanced,
            dp=1.2,
            param1=180,
            warm_param2=35,
            warm_min_r_frac=0.035,
            warm_max_r_frac=0.18,
            warm_min_dist_frac=0.10,
            sweep_param2=(30, 35, 40, 45, 50, 55, 60, 65),
            dedup=True,
            debug=False
        )
    raise ValueError(f"Unknown DETECT_METHOD_ID. Use 0..2.")