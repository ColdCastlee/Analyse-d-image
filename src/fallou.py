# fallou.py
import numpy as np
import cv2
import os
from core.data_loader import load_annotations
from core.evaluator import evaluate_dataset
from core.io_utils import imread_unicode
from core.preprocess import to_gray, denoise, enhance_contrast
from core.segmentation import apply_segmentation
from core.morphology import apply_morphology
from core.detection import detect_circles
from core.classification import classify_material_adaptive, estimate_values

IMG_PATH = "data/images/8.jpg"
def predict_pipeline_1(img_path):
    # adaptive + open + contour + ratio
    SEG_METHOD_ID = 1
    MORPH_METHOD_ID = 2
    DETECT_METHOD_ID = 1
    CLASSIFY_METHOD_ID = 1

    img = imread_unicode(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    gray = to_gray(img)
    blur = denoise(gray, (9, 9), 0)
    enhanced = enhance_contrast(blur)

    binary, _ = apply_segmentation(SEG_METHOD_ID, img, gray, enhanced)
    mask = apply_morphology(MORPH_METHOD_ID, binary, enhanced)

    circles = detect_circles(DETECT_METHOD_ID, img, enhanced, mask)
    if len(circles) == 0:
        return 0, 0.0

    materials = classify_material_adaptive(img, circles)
    _, cents, _ = estimate_values(CLASSIFY_METHOD_ID, img, circles, materials)

    total_cents = int(sum(int(v) for v in cents))
    return int(len(circles)), total_cents / 100.0


def predict_pipeline_2(img_path):
    # global Hough + ratio baseline
    CLASSIFY_METHOD_ID = 1

    img = imread_unicode(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    gray = to_gray(img)
    blur = denoise(gray, (9, 9), 2)

    circles_h = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=150,
        param1=100, param2=50,
        minRadius=60, maxRadius=200
    )

    if circles_h is None:
        return 0, 0.0

    circles_h = np.uint16(np.around(circles_h[0]))
    circles = [(int(x), int(y), float(r)) for (x, y, r) in circles_h]

    materials = ["unknown"] * len(circles)
    _, cents, _ = estimate_values(CLASSIFY_METHOD_ID, img, circles, materials)

    total_cents = int(sum(int(v) for v in cents))
    return int(len(circles)), total_cents / 100.0



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

IMAGES_DIR = os.path.join(ROOT_DIR, "data", "images")
ANN_PATH   = os.path.join(ROOT_DIR, "data", "annotations.csv")

def main():
    ann = load_annotations(ANN_PATH)

    evaluate_dataset(
        images_dir=IMAGES_DIR,
        ann_dict=ann,
        cfg={},  # unused when runner is provided
        runner=predict_pipeline_1,
        report_path=os.path.join(ROOT_DIR, "report_fallou_p1.txt"),
        tol_count=0,
        tol_euro=0.10
    )

    evaluate_dataset(
        images_dir=IMAGES_DIR,
        ann_dict=ann,
        cfg={},
        runner=predict_pipeline_2,
        report_path=os.path.join(ROOT_DIR, "report_fallou_p2.txt"),
        tol_count=0,
        tol_euro=0.10
    )

if __name__ == "__main__":
    main()