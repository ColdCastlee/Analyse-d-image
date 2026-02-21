# fallou.py
import os
import numpy as np
import cv2

from core.data_loader import load_annotations
from core.evaluator import evaluate_dataset, evaluate_one_image, _draw_circles
from core.io_utils import imread_unicode, debug_dump
from core.preprocess import to_gray, denoise
from core.segmentation import apply_segmentation
from core.detection import detect_circles
from core.classification import estimate_values

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src (nếu fallou.py nằm trong src)
ROOT_DIR = os.path.dirname(BASE_DIR)                   # .../Analyse-d-image

IMAGES_DIR = os.path.join(ROOT_DIR, "data", "images")
ANN_PATH   = os.path.join(ROOT_DIR, "data", "annotations.csv")

CFG = {
    "RUN_DEBUG_SINGLE": True,
    "DEBUG_MODE": "both",  # "none" | "show" | "save" | "both"
    "DEBUG_OUT_DIR": os.path.join(ROOT_DIR, "debug_out_fallou"),
    "DEBUG_IMAGE_PATH": os.path.join(IMAGES_DIR, "gp5", "18.jpg"),
    "WHICH": "both",       # "p1" | "p2" | "both"
}

# ----------------------------
# Helpers (tiny, matching original)
# ----------------------------
def _adaptive_threshold_like_original(blur):
    # original:
    # cv.adaptiveThreshold(blur, 255, GAUSSIAN_C, BINARY_INV, 31, 5)
    return cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 5
    )

def _opening_3x3_like_original(thresh):
    # original kernel = ones((3,3)), opening = MORPH_OPEN
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

def predict_pipeline_1(img_path):
    """
    Fallou P1 (original):
      gray -> GaussianBlur(9,9,0)
      adaptiveThreshold(31,5)
      opening 3x3
      contours -> minEnclosingCircle (radius>60)
      ratio classification
    """
    DETECT_METHOD_ID = 1          # contours + minEnclosingCircle
    CLASSIFY_METHOD_ID = 1        # ratio

    img = imread_unicode(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    debug_dump("P1__00_input", img, CFG, img_path)

    gray = to_gray(img)
    debug_dump("P1__01_gray", gray, CFG, img_path)

    blur = denoise(gray, (9, 9), 0)
    debug_dump("P1__02_blur", blur, CFG, img_path)

    # --- Use your segmentation module only as "label", but force original params ---
    # apply_segmentation(1) in your code uses default (11,2) -> NOT original.
    # So we do adaptiveThreshold with exact original params.
    thresh = _adaptive_threshold_like_original(blur)
    debug_dump("P1__03_thresh_adaptive31_5", thresh, CFG, img_path)

    opening = _opening_3x3_like_original(thresh)
    debug_dump("P1__04_opening3x3", opening, CFG, img_path)

    # detect circles from contours (same as your method_id=1)
    # NOTE: your detect_circles_contours_min_enclosing has min_radius_px=60 (same as original)
    circles = detect_circles(DETECT_METHOD_ID, img, blur, opening)
    debug_dump(f"P1__05_detect_circles_n{len(circles)}", _draw_circles(img, circles), CFG, img_path)

    if len(circles) == 0:
        debug_dump("P1__06_no_coins", img, CFG, img_path)
        return 0, 0.0

    # original doesn’t use materials; ratio only
    materials = ["unknown"] * len(circles)

    _, cents, _ = estimate_values(CLASSIFY_METHOD_ID, img, circles, materials)

    total_cents = int(sum(int(v) for v in cents))
    total_euros = total_cents / 100.0

    # value visualization (keep same style as your project)
    val_vis = img.copy()
    for i, (cx, cy, r) in enumerate(circles):
        v = int(cents[i]) if i < len(cents) else -1
        cv2.putText(
            val_vis, f"{i}:{v}c",
            (int(cx - r), int(cy + r)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2, cv2.LINE_AA
        )

    cv2.putText(
        val_vis, f"TOTAL: {total_euros:.2f} EUR ({total_cents}c)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    debug_dump("P1__06_values", val_vis, CFG, img_path)

    return int(len(circles)), float(total_euros)


def predict_pipeline_2(img_path):
    """
    Fallou P2 (original):
      gray -> GaussianBlur(9,9,2)
      HoughCircles(dp=1.2, minDist=150, param1=100, param2=50, minR=60, maxR=200)
      ratio classification
    """
    CLASSIFY_METHOD_ID = 1  # ratio

    img = imread_unicode(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    debug_dump("P2__00_input", img, CFG, img_path)

    gray = to_gray(img)
    debug_dump("P2__01_gray", gray, CFG, img_path)

    blur = denoise(gray, (9, 9), 2)
    debug_dump("P2__02_blur", blur, CFG, img_path)

    circles_h = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=150,
        param1=100, param2=50,
        minRadius=60, maxRadius=200
    )

    if circles_h is None:
        debug_dump("P2__03_no_circles", img, CFG, img_path)
        return 0, 0.0

    circles_h = np.uint16(np.around(circles_h[0]))
    circles = [(int(x), int(y), float(r)) for (x, y, r) in circles_h]
    debug_dump(f"P2__03_detect_circles_n{len(circles)}", _draw_circles(img, circles), CFG, img_path)

    materials = ["unknown"] * len(circles)
    _, cents, _ = estimate_values(CLASSIFY_METHOD_ID, img, circles, materials)

    total_cents = int(sum(int(v) for v in cents))
    total_euros = total_cents / 100.0

    val_vis = img.copy()
    for i, (cx, cy, r) in enumerate(circles):
        v = int(cents[i]) if i < len(cents) else -1
        cv2.putText(
            val_vis, f"{i}:{v}c",
            (int(cx - r), int(cy + r)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2, cv2.LINE_AA
        )

    cv2.putText(
        val_vis, f"TOTAL: {total_euros:.2f} EUR ({total_cents}c)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    debug_dump("P2__04_values", val_vis, CFG, img_path)

    return int(len(circles)), float(total_euros)


def main():
    ann = load_annotations(ANN_PATH)
    which = (CFG.get("WHICH") or "both").lower()

    if CFG.get("RUN_DEBUG_SINGLE", False):
        img_path = CFG["DEBUG_IMAGE_PATH"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(ROOT_DIR, img_path)

        if which in ("p1", "both"):
            evaluate_one_image(
                img_path=img_path,
                ann_dict=ann,
                cfg=CFG,
                tol_count=0,
                tol_euro=0.10,
                runner=predict_pipeline_1,
                name="Fallou_P1"
            )

        if which in ("p2", "both"):
            evaluate_one_image(
                img_path=img_path,
                ann_dict=ann,
                cfg=CFG,
                tol_count=0,
                tol_euro=0.10,
                runner=predict_pipeline_2,
                name="Fallou_P2"
            )

        mode = (CFG.get("DEBUG_MODE") or "none").lower()
        if mode in ("show", "both"):
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # batch
    if which in ("p1", "both"):
        evaluate_dataset(
            images_dir=IMAGES_DIR,
            ann_dict=ann,
            cfg=CFG,
            runner=predict_pipeline_1,
            report_path=os.path.join(ROOT_DIR, "report_fallou_p1.txt"),
            tol_count=0,
            tol_euro=0.10
        )

    if which in ("p2", "both"):
        evaluate_dataset(
            images_dir=IMAGES_DIR,
            ann_dict=ann,
            cfg=CFG,
            runner=predict_pipeline_2,
            report_path=os.path.join(ROOT_DIR, "report_fallou_p2.txt"),
            tol_count=0,
            tol_euro=0.10
        )

if __name__ == "__main__":
    main()