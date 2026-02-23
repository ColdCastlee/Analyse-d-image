# fallou.py
import os

from core.data_loader import load_annotations
from core.evaluator import evaluate_dataset, evaluate_one_image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src (nếu fallou.py nằm trong src)
ROOT_DIR = os.path.dirname(BASE_DIR)                   # .../Analyse-d-image

IMAGES_DIR = os.path.join(ROOT_DIR, "data", "images")
ANN_PATH   = os.path.join(ROOT_DIR, "data", "annotations.csv")
# fallou.py (example usage)
from core.io_utils import imread_unicode_v2, debug_dump_v2
from core.preprocess import to_gray_v2, denoise_v2, enhance_contrast_v2
from core.segmentation import apply_segmentation_v2
from core.morphology import apply_morphology_v2
from core.detection import detect_circles_hybrid_v2
from core.classification import classify_material_adaptive_v2, estimate_values_v2
from core.evaluator import _draw_circles_v2

def run_pipeline_v2(IMG_PATH, DEBUG_DIR=None):
    img = imread_unicode_v2(IMG_PATH)
    if img is None:
        raise FileNotFoundError(IMG_PATH)

    gray = to_gray_v2(img)
    gray = denoise_v2(gray, method="bilateral")
    gray = enhance_contrast_v2(gray, method="clahe")

    mask = apply_segmentation_v2(img, gray, mode="adaptive+edges")
    mask = apply_morphology_v2(mask, close_ksize=11, open_ksize=5, min_area=600)

    circles = detect_circles_hybrid_v2(img, gray, mask, hough_min_radius=10, hough_max_radius=150)

    materials = classify_material_adaptive_v2(img, circles)
    values, dbg = estimate_values_v2(img, circles, materials, euro_mode=True)

    vis = _draw_circles_v2(img, circles, values)

    if DEBUG_DIR:
        debug_dump_v2(DEBUG_DIR, "01_gray", gray)
        debug_dump_v2(DEBUG_DIR, "02_mask", mask)
        debug_dump_v2(DEBUG_DIR, "03_vis", vis, extra_text=str(dbg))

    return circles, materials, values, vis
CFG = {
    # dùng như fallou
    "RUN_DEBUG_SINGLE": False,
    "DEBUG_MODE": "none",  # batch sẽ bị evaluator ép về none anyway
    "DEBUG_OUT_DIR": os.path.join(ROOT_DIR, "debug_out_runner_new"),

    # nếu muốn chạy 1 ảnh debug
    "DEBUG_IMAGE_PATH": os.path.join(IMAGES_DIR, "gp4", "3.jpg"),

    # ==== config của bạn (tuỳ runner đọc gì) ====
    "SEG_METHOD_ID": 0,
    "MORPH_METHOD_ID": 6,
    "SEP_METHOD_ID": 1,
    "DETECT_METHOD_ID": 3,   # <-- NEW: 1=mask, 2=hough, 3=hybrid
    "CLASSIFY_METHOD_ID": 2,
}


def runner_new(img_path: str):
    circles, materials, values, vis = run_pipeline_v2(img_path, DEBUG_DIR=None)

    LABEL_TO_CENTS = {
        "2e": 200, "1e": 100,
        "50c": 50, "20c": 20, "10c": 10,
        "5c": 5, "2c": 2, "1c": 1
    }

    pred_count = int(len(circles))

    total_cents = 0
    for v in values:
        v = str(v).lower().replace("€", "e").strip()
        if v in LABEL_TO_CENTS:
            total_cents += LABEL_TO_CENTS[v]
        else:
            # nếu bạn có "small/mid/large" fallback trong estimate_values_v2
            # thì không cộng hoặc coi là 0
            total_cents += 0

    pred_euros = float(total_cents / 100.0)
    return pred_count, pred_euros

def main():
    ann = load_annotations(ANN_PATH)

    # ---------------- SINGLE DEBUG ----------------
    if CFG.get("RUN_DEBUG_SINGLE", False):
        img_path = CFG["DEBUG_IMAGE_PATH"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(ROOT_DIR, img_path)

        evaluate_one_image(
            img_path=img_path,
            ann_dict=ann,
            cfg=CFG,
            tol_count=0,
            tol_euro=0.10,
            runner=runner_new,          # <-- key line
            name="RUNNER_NEW"
        )
        return

    # ---------------- FULL DATA ----------------
    CFG["DEBUG_MODE"] = "none"  # keep like fallou

    evaluate_dataset(
        images_dir=IMAGES_DIR,
        ann_dict=ann,
        cfg=CFG,
        team_filter=None,
        tol_count=0,
        tol_euro=0.10,
        max_items=None,
        report_path=os.path.join(ROOT_DIR, "evaluation_report_runner_new.txt"),
        runner=runner_new          # <-- key line
    )


if __name__ == "__main__":
    main()