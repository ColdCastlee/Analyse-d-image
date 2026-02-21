# core/matching.py
import os
import re
import cv2
import numpy as np

LABEL_TO_CENTS = {
    "2€": 200, "1€": 100,
    "50c": 50, "20c": 20, "10c": 10,
    "5c": 5, "2c": 2, "1c": 1
}

def _normalize_label_from_filename(name: str):
    """
    Map your filenames:
      2e.jpg -> "2€"
      1e.jpg -> "1€"
      50c.jpg -> "50c"
      10c.jpg -> "10c"
      ...
    """
    base = os.path.splitext(os.path.basename(name))[0].lower().strip()
    base = base.replace(" ", "").replace("-", "").replace("_", "")

    # accept "2e", "1e" as euro coins
    if base == "2e":
        return "2€"
    if base == "1e":
        return "1€"

    # cents: "50c", "10c", ...
    m = re.fullmatch(r"(\d+)c", base)
    if m:
        v = int(m.group(1))
        if v in (50, 20, 10, 5, 2, 1):
            return f"{v}c"

    return None

def _read_ref_with_mask(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read ref: {path}")

    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        mask = (alpha > 0).astype(np.uint8) * 255
    else:
        bgr = img
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if (mask > 0).mean() > 0.8:
            mask = cv2.bitwise_not(mask)
    return bgr, mask

def _compute_area(mask_255):
    return int(np.count_nonzero(mask_255))

def _crop_tight(img, mask_255, pad=8):
    ys, xs = np.where(mask_255 > 0)
    if len(xs) == 0:
        return img, mask_255
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    h, w = mask_255.shape[:2]
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(w - 1, x1 + pad); y1 = min(h - 1, y1 + pad)
    return img[y0:y1+1, x0:x1+1], mask_255[y0:y1+1, x0:x1+1]

def build_ref_db(ref_dir, nfeatures=800):
    """
    Supports:
      A) ref_dir/2€/xxx.png ...
      B) ref_dir/2e.jpg, 50c.jpg ...
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)
    db = []

    # detect if "flat" (has image files directly)
    entries = [f for f in os.listdir(ref_dir) if os.path.isfile(os.path.join(ref_dir, f))]
    image_files = [f for f in entries if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]

    if len(image_files) > 0:
        # --- FLAT MODE ---
        for fn in sorted(image_files):
            if fn.lower().startswith("output"):
                continue
            label = _normalize_label_from_filename(fn)
            if label is None:
                continue
            path = os.path.join(ref_dir, fn)

            bgr, mask = _read_ref_with_mask(path)
            bgr, mask = _crop_tight(bgr, mask, pad=8)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            area = _compute_area(mask)

            kp, des = orb.detectAndCompute(gray, mask)

            db.append({
                "label": label,
                "cents": int(LABEL_TO_CENTS[label]),
                "bgr": bgr,
                "gray": gray,
                "mask": mask,
                "area": area,
                "kp": kp,
                "des": des
            })
    else:
        # --- SUBFOLDER MODE ---
        for label in sorted(os.listdir(ref_dir)):
            label_path = os.path.join(ref_dir, label)
            if not os.path.isdir(label_path):
                continue
            if label not in LABEL_TO_CENTS:
                continue
            for fn in sorted(os.listdir(label_path)):
                if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    continue
                path = os.path.join(label_path, fn)

                bgr, mask = _read_ref_with_mask(path)
                bgr, mask = _crop_tight(bgr, mask, pad=8)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                area = _compute_area(mask)

                kp, des = orb.detectAndCompute(gray, mask)

                db.append({
                    "label": label,
                    "cents": int(LABEL_TO_CENTS[label]),
                    "bgr": bgr,
                    "gray": gray,
                    "mask": mask,
                    "area": area,
                    "kp": kp,
                    "des": des
                })

    if len(db) == 0:
        raise RuntimeError(f"Ref DB empty. Check ref_dir={ref_dir} and filenames/labels.")
    return db

def _extract_coin_roi(img_bgr, mask_bin_255, circle, pad=20):
    """
    Extract ROI for a detected coin. Use circle as primary ROI,
    but also use global mask to estimate area more accurately.
    """
    cx, cy, r = circle
    H, W = mask_bin_255.shape[:2]
    cx, cy, r = int(cx), int(cy), int(round(r))

    x0 = max(0, cx - r - pad)
    y0 = max(0, cy - r - pad)
    x1 = min(W - 1, cx + r + pad)
    y1 = min(H - 1, cy + r + pad)

    roi = img_bgr[y0:y1+1, x0:x1+1].copy()
    roi_mask = mask_bin_255[y0:y1+1, x0:x1+1].copy()

    # additionally restrict mask to circle area to avoid neighbors
    hh, ww = roi_mask.shape[:2]
    yy, xx = np.ogrid[:hh, :ww]
    dist = np.sqrt((xx - (cx - x0))**2 + (yy - (cy - y0))**2)
    circle_mask = (dist <= r * 1.05).astype(np.uint8) * 255

    roi_mask = cv2.bitwise_and(roi_mask, circle_mask)
    return roi, roi_mask

def _good_match_score(des_q, kp_q, des_r, kp_r, use_ransac=True):
    """
    Compute matching score with ratio test.
    Optionally verify geometry using RANSAC homography.
    """
    if des_q is None or des_r is None:
        return 0.0, 0, 0

    if len(des_q) < 8 or len(des_r) < 8:
        return 0.0, 0, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des_q, des_r, k=2)

    good = []
    for m, n in knn:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        return float(len(good)), len(good), 0

    if not use_ransac:
        return float(len(good)), len(good), 0

    # RANSAC homography check (helps reject accidental matches)
    pts_q = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_r = np.float32([kp_r[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, inliers = cv2.findHomography(pts_q, pts_r, cv2.RANSAC, 5.0)
    if inliers is None:
        return float(len(good)) * 0.5, len(good), 0

    nin = int(inliers.sum())
    # score ưu tiên inliers
    score = nin + 0.2 * (len(good) - nin)
    return float(score), len(good), nin

def match_coin_orb_area(img_bgr, mask_bin_255, circle, ref_db,
                        area_tol=0.35, nfeatures=800, use_ransac=True):
    """
    CM06 matching coin:
    - Use area to filter candidates + estimate scale factor
    - ORB matching to pick best label

    area_tol=0.35 means accept ref with area ratio in [1-0.35, 1+0.35] after scale alignment.
    """

    roi_bgr, roi_mask = _extract_coin_roi(img_bgr, mask_bin_255, circle, pad=20)

    # area query
    area_q = _compute_area(roi_mask)
    if area_q < 200:
        return "unknown", 0.0

    # tight crop
    roi_bgr, roi_mask = _crop_tight(roi_bgr, roi_mask, pad=6)

    gray_q = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=nfeatures)

    best_label = "unknown"
    best_score = -1e9

    # Precompute keypoints of query at original scale once
    kp_q0, des_q0 = orb.detectAndCompute(gray_q, roi_mask)

    for ref in ref_db:
        area_r = ref["area"]
        if area_r < 200:
            continue

        # Estimate scale to align areas: area ~ s^2 → s = sqrt(area_r / area_q)
        s = float(np.sqrt(area_r / max(area_q, 1)))

        # If scale too extreme, skip early (stability)
        if s < 0.5 or s > 2.0:
            continue

        # Area-based filtering AFTER scale: scaled query area ~ area_q * s^2 ≈ area_r
        # Here we just enforce s not too far already; plus tolerance by ratio
        ratio = (area_q * (s ** 2)) / float(area_r)
        if abs(ratio - 1.0) > area_tol:
            continue

        # Resize query to ref-ish scale for better matching
        hq, wq = gray_q.shape[:2]
        new_w = max(32, int(round(wq * s)))
        new_h = max(32, int(round(hq * s)))

        gray_qs = cv2.resize(gray_q, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask_qs = cv2.resize(roi_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        kp_q, des_q = orb.detectAndCompute(gray_qs, mask_qs)

        score, n_good, n_in = _good_match_score(
            des_q, kp_q, ref["des"], ref["kp"], use_ransac=use_ransac
        )

        # small bonus if area alignment is tight
        score = score + 2.0 * (1.0 - abs(ratio - 1.0))

        if score > best_score:
            best_score = score
            best_label = ref["label"]

    return best_label, best_score