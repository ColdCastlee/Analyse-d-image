import cv2
import numpy as np

def _canny_auto(gray_u8, sigma=0.33):
    v = float(np.median(gray_u8))
    lo = int(max(0, (1.0 - sigma) * v))
    hi = int(min(255, (1.0 + sigma) * v))
    if hi <= lo + 5:
        lo = max(0, int(v * 0.5))
        hi = min(255, int(v * 1.5))
    return cv2.Canny(gray_u8, lo, hi)

def _arc_coverage_sample(center_x, center_y, r, hit_map_u8, step_deg=6, tol=2):
    """
    Estimate how much of the circumference is supported by hit_map (boundary/edges).
    Returns fraction in [0,1].
    """
    h, w = hit_map_u8.shape[:2]
    cx, cy = float(center_x), float(center_y)
    r = float(r)
    if r < 5:
        return 0.0

    hit = 0
    total = 0
    for ang in range(0, 360, step_deg):
        th = np.deg2rad(ang)
        x = int(round(cx + r * np.cos(th)))
        y = int(round(cy + r * np.sin(th)))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        total += 1

        # check small neighborhood tolerance
        x0, x1 = max(0, x - tol), min(w, x + tol + 1)
        y0, y1 = max(0, y - tol), min(h, y + tol + 1)
        if np.any(hit_map_u8[y0:y1, x0:x1] > 0):
            hit += 1

    return hit / max(total, 1)

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

def _dedup_concentric_keep_largest(circles, center_frac=0.18):
    """
    If two circles have (almost) same center -> keep the larger radius.
    Useful for 1€/2€ where inner ring edge is detected.
    """
    if not circles:
        return []
    circles = sorted(circles, key=lambda t: -t[2])  # keep larger first
    kept = []
    for (cx, cy, r) in circles:
        dup = False
        for (kx, ky, kr) in kept:
            if np.hypot(cx-kx, cy-ky) < center_frac * min(r, kr):
                dup = True
                break
        if not dup:
            kept.append((cx, cy, r))
    return kept

from scipy.ndimage import maximum_filter
import cv2
import numpy as np


def detect_circles_dist_localmax(
    enhanced, mask,
    min_radius_frac=0.5,
    peak_min_dist_frac=1.8,
    local_max_size=5,
    blur_sigma=3.0,
):
    """
    Peaks from distance-transform -> per-CC ROI -> local Hough -> filter by:
      - center gate to peak
      - overlap with mask
      - arc support on (mask boundary band) OR (image edges but ONLY near that band)
    This reduces "fake circles" from overlap regions / inner textures.
    """
    # Clean binary
    binary = (mask > 0).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

    # CC stats
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas_all = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    areas_all = areas_all[areas_all > 5000]
    if len(areas_all) == 0:
        print("No foreground after cleaning")
        return []

    r_estimates = np.sqrt(areas_all / np.pi)
    med_r = float(np.median(r_estimates))
    min_radius_px = max(30, min_radius_frac * med_r)
    peak_min_dist_px = max(50, peak_min_dist_frac * med_r)

    print(
        f"Adaptive: med_r={med_r:.1f} px | min_radius_px={min_radius_px:.1f} "
        f"| peak_min_dist_px={peak_min_dist_px:.1f} | clean CC={len(areas_all)}"
    )

    # Distance + blur
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    if blur_sigma > 0:
        dist = cv2.GaussianBlur(dist, (0, 0), sigmaX=blur_sigma)

    # Local maxima
    max_filtered = maximum_filter(dist, size=local_max_size)
    local_max = (dist == max_filtered)
    ys, xs = np.nonzero(local_max & (dist > min_radius_px))
    print(f"Raw local maxima count: {len(ys)}")
    if len(ys) == 0:
        print("No local maxima above min_radius_px")
        return []

    peaks = list(zip(xs, ys))
    values = dist[ys, xs]
    sorted_idx = np.argsort(-values)
    peaks = [peaks[i] for i in sorted_idx]
    values = values[sorted_idx]

    # NMS on peaks
    kept = []
    for i in range(len(peaks)):
        p = peaks[i]
        v = values[i]
        if any(np.hypot(p[0] - kp[0], p[1] - kp[1]) < peak_min_dist_px for kp, _ in kept):
            continue
        kept.append((p, v))
    print(f"Peaks after NMS: {len(kept)}")

    filtered_circles = []
    peak_gate = 0.85
    take_topk = 1

    for (cx, cy), r_peak in kept:
        if not np.isfinite(r_peak) or r_peak <= 0:
            continue

        label_id = labels[cy, cx]
        if label_id == 0:
            continue

        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]
        pad = int(0.3 * max(w, h))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(enhanced.shape[1], x + w + pad), min(enhanced.shape[0], y + h + pad)

        local_enhanced = enhanced[y0:y1, x0:x1].copy()
        local_mask = binary[y0:y1, x0:x1]

        # --- mask boundary band (stable) ---
        k_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dil = cv2.dilate(local_mask, k_edge, iterations=3)
        ero = cv2.erode(local_mask,  k_edge, iterations=3)
        band = cv2.subtract(dil, ero)
        band = (band > 0).astype(np.uint8) * 255

        # fill background outside mask to median FG to avoid Hough weirdness
        if np.any(local_mask > 0):
            fg_median = np.median(local_enhanced[local_mask > 0])
            local_enhanced[local_mask == 0] = fg_median

        local_blur = cv2.GaussianBlur(local_enhanced, (5, 5), 0)

        # --- edges, but only near band to avoid inner textures ---
        edges = _canny_auto(local_blur, sigma=0.33)
        edges = cv2.dilate(edges, k_edge, iterations=1)
        edges_on_band = cv2.bitwise_and(edges, band)

        # radius refs
        area_cc = float(stats[label_id, cv2.CC_STAT_AREA])
        r_cc = float(np.sqrt(area_cc / np.pi))
        cc_ratio = r_cc / max(r_peak, 1e-6)
        cc_is_single = (cc_ratio < 1.35)
        r_ref = r_peak if not cc_is_single else max(r_peak, 0.95 * r_cc)

        minR = int(max(10, 0.75 * r_ref))
        maxR = int(1.35 * r_ref)
        minDist = int(0.9 * r_ref)

        c = cv2.HoughCircles(
            local_blur, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=minDist,
            param1=110, param2=28,
            minRadius=minR, maxRadius=maxR
        )
        if c is None:
            continue

        c = np.squeeze(c)
        if c.ndim == 1:
            c = np.array([c])

        # peak in local coords
        pcx, pcy = cx - x0, cy - y0

        cands = []
        for hx, hy, hr in c:
            # 1) peak gate
            if np.hypot(hx - pcx, hy - pcy) > peak_gate * r_peak:
                continue

            # 2) overlap gate
            circle_mask = np.zeros(local_mask.shape, np.uint8)
            cv2.circle(circle_mask, (int(hx), int(hy)), int(hr), 255, -1)
            overlap_area = cv2.countNonZero(cv2.bitwise_and(circle_mask, local_mask))
            circle_area = np.pi * hr * hr
            overlap_frac = overlap_area / circle_area if circle_area > 0 else 0.0
            if overlap_frac < 0.5:
                continue

            # 3) arc support: mask-band is primary, edges-on-band is backup
            arc_m = _arc_coverage_sample(hx, hy, hr, band,         step_deg=6, tol=2)
            arc_e = _arc_coverage_sample(hx, hy, hr, edges_on_band, step_deg=6, tol=2)
            arc_support = max(arc_m, 0.85 * arc_e)

            # Soft gate (avoid killing true coins)
            if cc_is_single:
                min_arc = 0.12 if overlap_frac >= 0.60 else 0.16
            else:
                min_arc = 0.18 if overlap_frac >= 0.62 else 0.22

            # rescue: if overlap very high, allow smaller arc but not zero
            if not (overlap_frac >= 0.75 and arc_support >= 0.10):
                if arc_support < min_arc:
                    continue

            # Optional extra anti-fake: ring should mostly lie inside blob
            ring = np.zeros(local_mask.shape, np.uint8)
            cv2.circle(ring, (int(hx), int(hy)), int(hr), 255, thickness=2)
            inside = cv2.countNonZero(cv2.bitwise_and(ring, local_mask))
            total = cv2.countNonZero(ring)
            inside_frac = inside / max(total, 1)
            if inside_frac < 0.80:
                continue

            score = 0.70 * arc_support + 0.30 * overlap_frac - 0.12 * abs(hr - r_ref) / max(r_ref, 1e-6)
            cands.append((score, hx, hy, hr))

        if not cands:
            continue

        cands.sort(reverse=True)
        for _, hx, hy, hr in cands[:take_topk]:
            # promote inner -> outer for 1€/2€ (only when CC looks like single coin)
            if cc_is_single and hr < 0.78 * r_ref:
                hr2 = hr / 0.65
                hr = float(np.clip(hr2, 0.90 * r_ref, 1.35 * r_ref))

            filtered_circles.append((int(x0 + hx), int(y0 + hy), float(hr)))

    filtered_circles = _dedup_concentric_keep_largest(filtered_circles, center_frac=0.18)
    filtered_circles = _dedup_circles(filtered_circles, center_frac=0.42, r_frac=0.28)
    return filtered_circles

def detect_circles_cc_hough(img_bgr, enhanced, mask):
    circles = []

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas_all = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    if len(areas_all) == 0:
        return []

    med_area = float(np.median(areas_all))
    min_area = 0.4 * med_area
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
                param1=120, param2=24,
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

            # if no circles from Hough (previously had buggy dedup here - removed)

        # Fallback for small areas or failed Hough on large: use centroid + area-based radius
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


def detect_circles(method_id, img_bgr, enhanced, mask):
    """
    Returns circles as list of (cx, cy, r_px)
    """
    if method_id == 0:
        return detect_circles_cc_hough(img_bgr, enhanced, mask)
    if method_id == 1:
        return detect_circles_contours_min_enclosing(mask, min_radius_px=60)
    if method_id == 2: 
        return detect_circles_dist_localmax(
            enhanced, mask,
            min_radius_frac=0.6,        # 0.7 -> 0.6 (bắt thêm peak coin nhỏ)
            peak_min_dist_frac=1.4,     # 1.8 -> 1.4 (đừng NMS mạnh quá)
            local_max_size=9,           # 11 -> 9 (tạo thêm local maxima)
            blur_sigma=2.2              # 3.0 -> 2.2 (đỡ làm bẹt peak)
        )
    raise ValueError(f"Unknown DETECT_METHOD_ID={method_id}. Use 0..1.")