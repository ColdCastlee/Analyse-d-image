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

from scipy.ndimage import maximum_filter
import cv2
import numpy as np

def detect_circles_dist_localmax(mask, min_radius_frac=0.5, peak_min_dist_frac=1.8,
                                 local_max_size=9, blur_sigma=1.8):
    """
    FINAL FIX: tune for watershed over-segment.
    """
    # Clean binary (ignore boundaries)
    binary = (mask > 0).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)  # Reduced to 1
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Adaptive from clean binary
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas_all = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    areas_all = areas_all[areas_all > 5000]  # Increased to filter tiny
    if len(areas_all) == 0:
        print("No foreground after cleaning")
        return []
    r_estimates = np.sqrt(areas_all / np.pi)
    med_r = float(np.median(r_estimates))
    min_radius_px = max(30, min_radius_frac * med_r)
    peak_min_dist_px = max(50, peak_min_dist_frac * med_r)

    print(f"Adaptive: med_r={med_r:.1f} px | min_radius_px={min_radius_px:.1f} | peak_min_dist_px={peak_min_dist_px:.1f} | clean CC={len(areas_all)}")

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

    # NMS
    kept = []
    for i in range(len(peaks)):
        p = peaks[i]
        v = values[i]
        if any(np.hypot(p[0]-kp[0], p[1]-kp[1]) < peak_min_dist_px for kp, _ in kept):
            continue
        kept.append((p, v))

    print(f"Peaks after NMS: {len(kept)}")

    # Circularity on clean binary (crop)
    filtered_circles = []
    for (cx, cy), r in kept:
        if not np.isfinite(r) or r <= 0:
            continue
        pad = int(2.0 * r)
        x0, y0 = max(0, cx - pad), max(0, cy - pad)
        x1, y1 = min(binary.shape[1], cx + pad), min(binary.shape[0], cy + pad)
        local = binary[y0:y1, x0:x1]

        contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if area < 5000:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        print(f"  Peak ({cx},{cy}) r≈{r:.1f} → circularity={circularity:.3f} area={area:.0f}")
        if circularity > 0.18:  # Lowered to keep slight merges
            filtered_circles.append((int(cx), int(cy), float(r)))

    filtered_circles = _dedup_circles(filtered_circles, center_frac=0.42, r_frac=0.28)

    print(f"FINAL detected coins: {len(filtered_circles)}")
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
        return detect_circles_dist_localmax(mask, min_radius_frac=0.65, peak_min_dist_frac=1.65,
                                        local_max_size=9, blur_sigma=1.8)
    raise ValueError(f"Unknown DETECT_METHOD_ID={method_id}. Use 0..1.")