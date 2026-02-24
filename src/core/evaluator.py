import math
import cv2
import os

from core.io_utils import imread_unicode, show_fit, debug_dump
from core.preprocess import to_gray, denoise, enhance_contrast
from core.segmentation import apply_segmentation
from core.morphology import apply_morphology
from core.separation import watershed_separate
from core.detection import detect_circles
from core.classification import classify_material_adaptive, estimate_values

from core.data_loader import iter_image_files, basename_only


def mae(xs):
    return sum(abs(x) for x in xs) / max(1, len(xs))

def rmse(xs):
    return math.sqrt(sum((x * x) for x in xs) / max(1, len(xs)))


def _draw_circles(img_bgr, circles):
    out = img_bgr.copy()
    for (cx, cy, r) in circles:
        cv2.circle(out, (int(cx), int(cy)), int(round(r)), (0, 255, 0), 2)
        cv2.circle(out, (int(cx), int(cy)), 2, (0, 0, 255), -1)
    return out

def run_pipeline_on_image(img_path, cfg):
    """
    Returns: (pred_count:int, pred_euros:float)
    """
    img = imread_unicode(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    debug_dump("00_input", img, cfg, img_path)

    gray = to_gray(img)

    blur = denoise(gray, (7, 7), 0)
    
    enhanced = enhance_contrast(blur, clip=2.0, grid=(8, 8))

    binary, seg_name = apply_segmentation(cfg["SEG_METHOD_ID"], img, gray, enhanced)
    debug_dump(f"04_binary_{seg_name}", binary, cfg, img_path)

    mask = apply_morphology(cfg["MORPH_METHOD_ID"], binary, enhanced)
    debug_dump(f"05_mask_morph{cfg['MORPH_METHOD_ID']}", mask, cfg, img_path)

    if cfg["SEP_METHOD_ID"] == 1:
        mask = watershed_separate(img, mask, show_debug=False, show_fit=show_fit)
        debug_dump("06_mask_watershed", mask, cfg, img_path)

    circles = detect_circles(cfg["DETECT_METHOD_ID"], img, enhanced, mask)
    debug_dump(f"07_detect_circles_n{len(circles)}", _draw_circles(img, circles), cfg, img_path)

    pred_count = int(len(circles))
    if pred_count == 0:
        # Optional: still dump a "no coins" stage
        debug_dump("08_no_coins", img, cfg, img_path)
        return 0, 0.0

    # ---------------------------------------------------------
    # 08) Material classification debug (overlay labels)
    # ---------------------------------------------------------
    materials = classify_material_adaptive(img, circles)

    mat_vis = img.copy()
    for i, (cx, cy, r) in enumerate(circles):
        m = materials[i] if i < len(materials) else "unknown"
        cv2.putText(
            mat_vis,
            f"{i}:{m}",
            (int(cx - r), int(cy - r)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    debug_dump("08_materials", mat_vis, cfg, img_path)

    # ---------------------------------------------------------
    # 09) Value estimation debug
    # ---------------------------------------------------------
    if cfg["CLASSIFY_METHOD_ID"] == 2:
        _, cents_list, _ = estimate_values(
            cfg["CLASSIFY_METHOD_ID"], img, circles, materials,
            mask_bin_255=mask
        )
    else:
        _, cents_list, _ = estimate_values(
            cfg["CLASSIFY_METHOD_ID"], img, circles, materials
        )

    total_cents = int(sum(int(v) for v in cents_list))
    total_euros = total_cents / 100.0

    val_vis = img.copy()
    for i, (cx, cy, r) in enumerate(circles):
        v = int(cents_list[i]) if i < len(cents_list) else -1
        text = f"{i}:{v}c"
        x = int(cx - r)
        y = int(cy + r + 30)

        # --- compute text size ---
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)

        # --- draw background rectangle (black) ---
        cv2.rectangle(val_vis, (x - 6, y - h - 6), (x + w + 6, y + 6), (0, 0, 0), -1)

        # --- draw text (white) ---
        cv2.putText(val_vis, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # show total on top-left
    total_text = f"TOTAL: {total_euros:.2f} EUR ({total_cents}c)"

    # --- compute text size ---
    (w, h), _ = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)

    # --- draw background rectangle (black) ---
    cv2.rectangle(val_vis, (15, 15), (20 + w + 10, 50 + h), (0, 0, 0), -1)

    # --- draw text (white) ---
    cv2.putText(val_vis, total_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    debug_dump("09_values", val_vis, cfg, img_path)

    return pred_count, total_euros


def _safe_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def write_report_txt(report_path, summary_lines, mismatch_lines, error_lines, missing_lines):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== SUMMARY ===\n")
        for line in summary_lines:
            f.write(line.rstrip() + "\n")

        f.write("\n=== MISMATCHES (all) ===\n")
        if len(mismatch_lines) == 0:
            f.write("None\n")
        else:
            for line in mismatch_lines:
                f.write(line.rstrip() + "\n")

        f.write("\n=== FAILURES / EXCEPTIONS ===\n")
        if len(error_lines) == 0:
            f.write("None\n")
        else:
            for line in error_lines:
                f.write(line.rstrip() + "\n")

        f.write("\n=== MISSING ANNOTATIONS ===\n")
        if len(missing_lines) == 0:
            f.write("None\n")
        else:
            for line in missing_lines:
                f.write(line.rstrip() + "\n")

def evaluate_one_image(img_path, ann_dict, cfg, tol_count=0, tol_euro=0.10, runner=None, name=None):
    """
    Evaluate a single image (per-image metrics only; no aggregates).

    runner: callable(img_path)->(pred_count:int, pred_euros:float)
            If None, use run_pipeline_on_image(img_path, cfg).

    name: optional label printed in console (e.g., "Fallou_P1").
    """
    team = os.path.basename(os.path.dirname(img_path))
    fn = basename_only(img_path)
    key = (team, fn)

    if key not in ann_dict:
        print(f"[SKIP] {team}/{fn} -> annotation not found")
        return

    gt_count = ann_dict[key].get("count", None)
    gt_euros = ann_dict[key].get("euros", None)

    if gt_count is None or gt_euros is None:
        print(f"[SKIP] {team}/{fn} -> incomplete annotation: count={gt_count}, euros={gt_euros}")
        return

    try:
        if runner is None:
            pred_count, pred_euros = run_pipeline_on_image(img_path, cfg)
        else:
            pred_count, pred_euros = runner(img_path)

        pred_count = int(pred_count)
        pred_euros = float(pred_euros)

        ce = int(pred_count - int(gt_count))
        ve = float(pred_euros - float(gt_euros))

        ok_count = (abs(ce) <= tol_count)
        ok_euro  = (abs(ve) <= tol_euro)
        ok_both  = ok_count and ok_euro

        tag = f"{name} | " if name else ""

        print("=================================================")
        print(f"{tag}IMAGE: {team}/{fn}")
        print(f"GT   : count={int(gt_count)}  euros={float(gt_euros):.2f}")
        print(f"PRED : count={int(pred_count)} euros={float(pred_euros):.2f}")
        print("-------------------------------------------------")
        print(f"[COUNT] err={ce:+d}  pass(|err|<={tol_count})={ok_count}")
        print(f"[EURO ] err={ve:+.2f}€ pass(|err|<={tol_euro:.2f}€)={ok_euro}")
        print(f"[BOTH ] pass={ok_both}")
        print("=================================================")

        # If using built-in pipeline, debug windows may be open -> wait for keypress
        mode = (cfg.get("DEBUG_MODE") or "none").lower()
        if runner is None and mode in ("show", "both"):
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    except Exception as e:
        tag = f"{name} | " if name else ""
        print("=================================================")
        print(f"{tag}IMAGE: {team}/{fn}")
        print(f"GT   : count={gt_count} euros={_safe_float(gt_euros):.2f}")
        print(f"ERROR: {e}")
        print("=================================================")

def evaluate_dataset(images_dir, ann_dict, cfg,
                     team_filter=None,
                     tol_count=0, tol_euro=0.10,
                     max_items=None,
                     report_path="evaluation_report.txt",
                     runner=None):
    """
    Batch evaluation.
    IMPORTANT: Debug visualization/saving is disabled in batch mode
    to avoid generating too many images/windows.
    """

    def _safe_float_local(x, default=0.0):
        try:
            if x is None:
                return float(default)
            return float(x)
        except Exception:
            return float(default)

    paths = list(iter_image_files(images_dir))

    if team_filter is not None:
        paths = [p for p in paths if os.sep + team_filter + os.sep in p]

    paths = sorted(paths)
    if max_items is not None:
        paths = paths[:max_items]

    results_ok = []
    results_err = []
    missing_lines = []
    missing_ann = 0

    # detection success/failure (by "found at least 1 coin")
    success_detect = 0
    fail_detect = 0
    exception_count = 0

    for p in paths:
        team = os.path.basename(os.path.dirname(p))  # .../images/gp5/xxx.jpg
        fn = basename_only(p)
        key = (team, fn)

        # --- annotation check ---
        if key not in ann_dict:
            missing_ann += 1
            missing_lines.append(f"{team}/{fn} -> annotation not found")
            continue

        gt_count = ann_dict[key].get("count", None)
        gt_euros = ann_dict[key].get("euros", None)

        # Skip if annotation incomplete (do NOT run pipeline)
        if gt_count is None or gt_euros is None:
            missing_ann += 1
            missing_lines.append(
                f"{team}/{fn} -> skipped (incomplete annotation: count={gt_count}, euros={gt_euros})"
            )
            continue

        # --- run pipeline once ---
        try:
            if runner is None:
                pred_count, pred_euros = run_pipeline_on_image(p, cfg)
            else:
                pred_count, pred_euros = runner(p)

            pred_count = int(pred_count)
            pred_euros = float(pred_euros)

            # detection success/fail definition
            if pred_count > 0:
                success_detect += 1
            else:
                fail_detect += 1

            results_ok.append({
                "team": team, "file": fn,
                "gt_count": int(gt_count),
                "gt_euros": float(gt_euros),
                "pred_count": pred_count,
                "pred_euros": pred_euros,
            })

        except Exception as e:
            exception_count += 1
            results_err.append({
                "team": team, "file": fn,
                "gt_count": gt_count,
                "gt_euros": gt_euros,
                "err": str(e)
            })

    evaluated_images = len(results_ok)  # only those with annotations & no exception

    if evaluated_images == 0:
        print("No valid results to evaluate.")
        print("Missing annotations:", missing_ann)
        print("Exceptions:", exception_count)

        summary_lines = [
            f"images_dir: {images_dir}",
            f"team_filter: {team_filter}",
            "evaluated_images (no exception): 0",
            f"missing_annotations: {missing_ann}",
            f"exceptions: {exception_count}",
        ]

        error_lines = [
            f"{r['team']}/{r['file']} | GT count={r['gt_count']} GT €={_safe_float_local(r['gt_euros']):.2f} | ERROR: {r['err']}"
            for r in results_err
        ]

        # If your evaluator.py already has write_report_txt(), keep using it.
        # Otherwise, remove these 2 lines.
        write_report_txt(report_path, summary_lines, [], error_lines, missing_lines)
        print(f"Report saved to: {report_path}")
        return

    # Compute errors for metrics
    count_errs = [(r["pred_count"] - r["gt_count"]) for r in results_ok]
    euro_errs  = [(r["pred_euros"] - r["gt_euros"]) for r in results_ok]

    count_mae = mae(count_errs)
    count_rmse = rmse(count_errs)
    count_acc = sum(1 for e in count_errs if abs(e) <= tol_count) / len(count_errs)

    euro_mae = mae(euro_errs)
    euro_rmse = rmse(euro_errs)
    euro_acc = sum(1 for e in euro_errs if abs(e) <= tol_euro) / len(euro_errs)

    # Correct image counts (ground-truth based)
    correct_count_images = sum(1 for r in results_ok if r["pred_count"] == r["gt_count"])
    correct_money_images = sum(1 for r in results_ok if abs(r["pred_euros"] - r["gt_euros"]) <= tol_euro)
    correct_both_images = sum(
        1 for r in results_ok
        if (r["pred_count"] == r["gt_count"]) and (abs(r["pred_euros"] - r["gt_euros"]) <= tol_euro)
    )

    # Build mismatch lines (ALL)
    mismatch_lines = []
    for r in results_ok:
        ce = int(r["pred_count"] - r["gt_count"])
        ve = float(r["pred_euros"] - r["gt_euros"])
        wrong_count = (r["pred_count"] != r["gt_count"])
        wrong_value = (abs(ve) > tol_euro)

        if wrong_count or wrong_value:
            mismatch_lines.append(
                f"{r['team']}/{r['file']} | "
                f"GT count={r['gt_count']} pred={r['pred_count']} err={ce:+d} | "
                f"GT €={r['gt_euros']:.2f} pred={r['pred_euros']:.2f} err={ve:+.2f} | "
                f"value_tol=±{tol_euro:.2f}"
            )

    # Build error lines (ALL exceptions)
    error_lines = []
    for r in results_err:
        error_lines.append(
            f"{r['team']}/{r['file']} | "
            f"GT count={r['gt_count']} GT €={_safe_float_local(r['gt_euros']):.2f} | "
            f"ERROR: {r['err']}"
        )

    # Summary lines (for txt)
    summary_lines = [
        f"images_dir: {images_dir}",
        f"team_filter: {team_filter}",
        f"evaluated_images (no exception): {evaluated_images}",
        f"missing_annotations: {missing_ann}",
        f"exceptions: {exception_count}",
        "",
        f"detection_success (pred_count>0): {success_detect}/{evaluated_images}",
        f"detection_fail (pred_count==0): {fail_detect}/{evaluated_images}",
        "",
        f"[COUNT] MAE={count_mae:.3f} RMSE={count_rmse:.3f} Accuracy(|err|<={tol_count})={count_acc*100:.1f}%",
        f"[EURO ] MAE={euro_mae:.3f} RMSE={euro_rmse:.3f} Accuracy(|err|<={tol_euro:.2f}€)={euro_acc*100:.1f}%",
        "",
        f"Correct coin count images: {correct_count_images}/{evaluated_images} ({100.0*correct_count_images/evaluated_images:.1f}%)",
        f"Correct monetary value images: {correct_money_images}/{evaluated_images} ({100.0*correct_money_images/evaluated_images:.1f}%) (tolerance ±{tol_euro:.2f} €)",
        f"Correct BOTH (count+value): {correct_both_images}/{evaluated_images} ({100.0*correct_both_images/evaluated_images:.1f}%)",
        "",
        f"mismatch_cases: {len(mismatch_lines)}",
        f"error_cases: {len(error_lines)}",
        f"missing_or_skipped_cases: {len(missing_lines)}",
    ]

    # Print short summary to console (English)
    print("=================================================")
    print("EVALUATION")
    print("images_dir:", images_dir)
    print("team_filter:", team_filter)
    print(f"evaluated images (no exception): {evaluated_images}")
    print(f"missing annotations: {missing_ann} | exceptions: {exception_count}")
    print("-------------------------------------------------")
    print(f"Detection success (pred_count>0): {success_detect}/{evaluated_images}")
    print(f"Detection fail    (pred_count==0): {fail_detect}/{evaluated_images}")
    print("-------------------------------------------------")
    print(f"[COUNT] MAE={count_mae:.3f}  RMSE={count_rmse:.3f}  Accuracy(|err|<={tol_count})={count_acc*100:.1f}%")
    print(f"[EURO ] MAE={euro_mae:.3f}  RMSE={euro_rmse:.3f}  Accuracy(|err|<={tol_euro:.2f}€)={euro_acc*100:.1f}%")
    print("-------------------------------------------------")
    print(f"Correct coin count images     : {correct_count_images}/{evaluated_images}")
    print(f"Correct monetary value images : {correct_money_images}/{evaluated_images} (±{tol_euro:.2f}€)")
    print(f"Correct BOTH (count + value)  : {correct_both_images}/{evaluated_images}")
    print("=================================================")

    # Write report (requires write_report_txt to exist in this file)
    write_report_txt(report_path, summary_lines, mismatch_lines, error_lines, missing_lines)
    print(f"Report saved to: {report_path}")