# src/main.py
import os

from core.data_loader import load_annotations
from core.evaluator import evaluate_dataset, evaluate_one_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src
ROOT_DIR = os.path.dirname(BASE_DIR)                   # .../Analyse-d-image

IMAGES_DIR = os.path.join(ROOT_DIR, "data", "images")
ANN_PATH   = os.path.join(ROOT_DIR, "data", "annotations.csv")

# =========================================================
# PIPELINE CONFIGURATION (edit here)
# =========================================================
CFG = {
    # -----------------------------------------------------
    # SEGMENTATION METHOD
    # 0 = Otsu thresholding (global)
    # 1 = Adaptive thresholding (local)
    # 2 = MSER-based region detection
    # 3 = K-means color clustering
    # -----------------------------------------------------
    "SEG_METHOD_ID": 4,

    # -----------------------------------------------------
    # MORPHOLOGICAL POST-PROCESSING
    # 0 = Dilate
    # 1 = Erode
    # 2 = Open
    # 3 = Close
    # 4 = Gray close + Otsu
    # 5 = Cascade (erode -> dilate)
    # 6 = Median filter on closed mask (recommended)
    # 7 = Majority filter
    # -----------------------------------------------------
    "MORPH_METHOD_ID": 6,

    # -----------------------------------------------------
    # SEPARATION (for overlapping coins)
    # 0 = No separation
    # 1 = Watershed separation
    # -----------------------------------------------------
    "SEP_METHOD_ID": 1,

    # -----------------------------------------------------
    # COIN DETECTION METHOD
    # 0 = Connected Components + local Hough (robust, default)
    # 1 = Contours + minEnclosingCircle (fast baseline)
    # -----------------------------------------------------
    "DETECT_METHOD_ID": 2,

    # -----------------------------------------------------
    # COIN CLASSIFICATION / VALUE ESTIMATION
    # 0 = Bimetal-based scale estimation (mm anchor)
    #     -> requires detecting at least one 1€ or 2€ coin
    #
    # 1 = Radius ratio method (Fallou baseline)
    #     -> uses relative size only, no real-world scale
    #
    # 2 = CM06 feature matching (ORB + RANSAC + area)
    #     -> matches detected coins with reference images
    #     -> fallback to mm-based method if matching fails
    #     -> requires reference images in data/ref/
    # -----------------------------------------------------
    "CLASSIFY_METHOD_ID": 2,

    # =====================================================
    # RUN MODE (CHOOSE ONE)
    # =====================================================
    # True  -> single-image debug mode
    #          (image visualization / saving is allowed)
    # False -> batch evaluation mode
    #          (DEBUG_MODE will be forced to "none")
    "RUN_DEBUG_SINGLE": False,

    # Debug output mode
    # Effective ONLY when RUN_DEBUG_SINGLE = True
    # Options:
    #   "none" -> no visualization, no saving
    #   "show" -> display images only
    #   "save" -> save images only
    #   "both" -> display and save images
    "DEBUG_MODE": "save",

    # Output directory for saved debug images
    "DEBUG_OUT_DIR": os.path.join(ROOT_DIR, "debug_out_gp8_n"),

    # Image path for single-image debug
    # Used ONLY when RUN_DEBUG_SINGLE = True
    "DEBUG_IMAGE_PATH": os.path.join(IMAGES_DIR, "gp4", "3.jpg"),

}
def main():
    TEAM = "gp8"
    ann = load_annotations(ANN_PATH, team_filter=TEAM)

    # -----------------------------------------------------
    # SINGLE-IMAGE DEBUG MODE
    # -----------------------------------------------------
    if CFG.get("RUN_DEBUG_SINGLE", False):
        img_path = CFG["DEBUG_IMAGE_PATH"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(ROOT_DIR, img_path)

        evaluate_one_image(
            img_path=img_path,
            ann_dict=ann,
            cfg=CFG,
            tol_count=0,
            tol_euro=0.10
        )
        return

    # -----------------------------------------------------
    # BATCH EVALUATION MODE
    # -----------------------------------------------------
    # Force DEBUG_MODE to "none" to avoid excessive
    # visualization or disk usage during batch processing
    # CFG["DEBUG_MODE"] = "none"

    evaluate_dataset(
        images_dir=IMAGES_DIR,
        ann_dict=ann,
        cfg=CFG,
        team_filter=TEAM,
        tol_count=0,
        tol_euro=0.10,
        max_items=None,
        report_path=os.path.join(ROOT_DIR, "evaluation_report_gp5.txt")
    )

if __name__ == "__main__":
    main()