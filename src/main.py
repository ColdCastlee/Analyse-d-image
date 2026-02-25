# src/main.py
import os
from core.data_loader import load_annotations
from core.evaluator import evaluate_dataset, evaluate_one_image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGES_DIR = os.path.join(ROOT_DIR, "data", "images")
ANN_PATH = os.path.join(ROOT_DIR, "data", "annotations.csv")

CFG = {
    # evaluator + pure-hough thực sự dùng
    "DETECT_METHOD_ID": 2,

    # debug control
    "RUN_DEBUG_SINGLE": False,
    "DEBUG_MODE": "none",
    "DEBUG_OUT_DIR": os.path.join(ROOT_DIR, "debug_new_gp1_7"),
    "DEBUG_IMAGE_PATH": os.path.join(IMAGES_DIR, "gp1", "18.png"),
}

TOL_COUNT = 0
TOL_EURO = 0.10


def main():
    team = None
    ann = load_annotations(ANN_PATH)

    if CFG["RUN_DEBUG_SINGLE"]:
        evaluate_one_image(
            img_path=CFG["DEBUG_IMAGE_PATH"],
            ann_dict=ann,
            cfg=CFG,
            tol_count=TOL_COUNT,
            tol_euro=TOL_EURO,
        )
        return

    evaluate_dataset(
        team_filter = team,
        images_dir=IMAGES_DIR,
        ann_dict=ann,
        cfg=CFG,
        tol_count=TOL_COUNT,
        tol_euro=TOL_EURO,
        report_path=os.path.join(ROOT_DIR, "evaluation_report.txt"),
    )


if __name__ == "__main__":
    main()