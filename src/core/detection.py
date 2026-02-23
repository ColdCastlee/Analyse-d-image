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
    min_area = 0.4 * med_area
    split_area_th = 1.58 * med_area

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
            minR = int(max(10, 0.6 * r_guess))
            maxR = int(1.45 * r_guess)
            minDist = int(max(20, 1.0 * r_guess))

            c = cv2.HoughCircles(
                roi_blur, cv2.HOUGH_GRADIENT,
                dp=1.2, minDist=minDist,
                param1=100, param2=25,
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

                # Keep up to 3 circles from Hough candidates
                kept = sorted(kept, key=lambda t: -t[2])[:4]

                if len(kept) >= 1:
                    cx1, cy1, r1 = kept[0]
                    circles.append((int(x0 + cx1), int(y0 + cy1), float(r1)))
                    for k in range(1, len(kept)):
                        cxk, cyk, rk = kept[k]
                        dist = float(np.hypot(cx1 - cxk, cy1 - cyk))
                        if dist > 1.2 * min(r1, rk):  # Lower thresh to accept more
                            circles.append((int(x0 + cxk), int(y0 + cyk), float(rk)))
                    continue
            cx, cy = centroids[i]
            r = float(np.sqrt(area / np.pi))
            circles.append((int(cx), int(cy), r))
            continue

        cx, cy = centroids[i]
        r = float(np.sqrt(area / np.pi))
        circles.append((int(cx), int(cy), r))

    # Remove duplicate detections (same coin detected multiple times)
    before = len(circles)
    circles = _dedup_circles(circles, center_frac=0.3, r_frac=0.3)
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
    raise ValueError(f"Unknown DETECT_METHOD_ID={method_id}. Use 0..1.")