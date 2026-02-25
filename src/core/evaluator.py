import math
import cv2
import os
import numpy as np
from core.io_utils import imread_unicode, show_fit, debug_dump
from core.data_loader import iter_image_files, basename_only
from core.classification import Circle, classify_euro_coins

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

def to_gray(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def denoise(gray, ksize=(7, 7), sigma=0):
    return cv2.GaussianBlur(gray, ksize, sigma)

def enhance_contrast(gray_blur, clip=2.0, grid=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(gray_blur)

def detect_circles_hough_pure(enhanced,
                              dp=1.2,
                              min_dist_frac=0.08,
                              param1=120,
                              param2=28,
                              min_r_frac=0.04,
                              max_r_frac=0.20,
                              dedup=True):
    """
    PURE HOUGH on preprocessed image (enhanced = gray+blur+CLAHE).
    No mask, no CC, no DT.
    """
    if enhanced is None:
        return []
    h, w = enhanced.shape[:2]
    min_dim = min(h, w)
    minRadius = max(8, int(min_dim * min_r_frac))
    maxRadius = max(minRadius + 2, int(min_dim * max_r_frac))
    minDist = max(12, int(min_dim * min_dist_frac))
    circles = cv2.HoughCircles(
        enhanced,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )
    if circles is None:
        return []
    circles = np.round(circles[0]).astype(int)
    circles = [(int(x), int(y), int(r)) for (x, y, r) in circles]
    if not dedup:
        return circles
    circles = sorted(circles, key=lambda t: -t[2])
    kept = []
    for (x, y, r) in circles:
        dup = False
        for (kx, ky, kr) in kept:
            if np.hypot(x - kx, y - ky) < 0.20 * min(r, kr):
                dup = True
                break
        if not dup:
            kept.append((x, y, r))
    return kept

def run_pipeline_on_image(img_path, cfg):
    debug_mode = str(cfg.get("DEBUG_MODE", "off")).lower()
    if debug_mode in ("save", "both"):
        out_dir = cfg.get("DEBUG_OUT_DIR", "debug_out")
        os.makedirs(out_dir, exist_ok=True)
        print(f"[DEBUG] Created/Checked dir: {out_dir}")

    img = imread_unicode(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")

    debug_dump("00_input", img, cfg, img_path)

    gray = to_gray(img)
    blur = denoise(gray, (7, 7), 0)
    enhanced = enhance_contrast(blur, clip=2.0, grid=(8, 8))

    circles = detect_circles_hough_pure(
        enhanced,
        dp=1.2,
        min_dist_frac=0.165,
        param1=220,
        param2=60,
        min_r_frac=0.04,
        max_r_frac=0.13,
        dedup=True
    )

    debug_dump(f"04_detect_circles_n{len(circles)}", _draw_circles(img, circles), cfg, img_path)

    pred_count = len(circles)
    if pred_count == 0:
        debug_dump("05_no_coins", img, cfg, img_path)
        return 0, 0.0

    # ---- classification.py integration (IMPORTANT) ----
    circles_cls = [Circle(float(cx), float(cy), float(r)) for (cx, cy, r) in circles]
    results = classify_euro_coins(img, circles=circles_cls, ref_db=None)

    cents_list = [r.cents for r in results]
    materials  = [r.material for r in results]
    scale = results[0].scale_px_per_mm if results and results[0].scale_px_per_mm is not None else None

    print(f"[DEBUG] scale={scale} materials={materials}")
    print(f"[DEBUG] cents_list len={len(cents_list)} values={cents_list}")

    total_cents = int(sum(int(v) for v in cents_list))
    total_euros = total_cents / 100.0

    # ---- visualization (safe) ----
    try:
        val_vis = img.copy()
        for i, (cx, cy, r) in enumerate(circles):
            v = int(cents_list[i]) if i < len(cents_list) else -1
            text = f"{i}:{v}c"
            x = int(cx - r)
            y = int(cy + r + 30)
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(val_vis, (x - 6, y - h - 6), (x + w + 6, y + 6), (0, 0, 0), -1)
            cv2.putText(val_vis, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        total_text = f"TOTAL: {total_euros:.2f} EUR ({total_cents}c)"
        (w, h), _ = cv2.getTextSize(total_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.rectangle(val_vis, (15, 15), (20 + w + 10, 50 + h), (0, 0, 0), -1)
        cv2.putText(val_vis, total_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

        debug_dump("07_values", val_vis, cfg, img_path)
    except Exception as e:
        print(f"[WARN] val_vis skipped: {e} for {img_path}")

    return int(pred_count), float(total_euros)

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
        ok_euro = (abs(ve) <= tol_euro)
        ok_both = ok_count and ok_euro
        tag = f"{name} | " if name else ""
        print("=================================================")
        print(f"{tag}IMAGE: {team}/{fn}")
        print(f"GT : count={int(gt_count)} euros={float(gt_euros):.2f}")
        print(f"PRED : count={int(pred_count)} euros={float(pred_euros):.2f}")
        print("-------------------------------------------------")
        print(f"[COUNT] err={ce:+d} pass(|err|<={tol_count})={ok_count}")
        print(f"[EURO ] err={ve:+.2f}€ pass(|err|<={tol_euro:.2f}€)={ok_euro}")
        print(f"[BOTH ] pass={ok_both}")
        print("=================================================")
        mode = (cfg.get("DEBUG_MODE") or "none").lower()
        if runner is None and mode in ("show", "both"):
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        tag = f"{name} | " if name else ""
        print("=================================================")
        print(f"{tag}IMAGE: {team}/{fn}")
        print(f"GT : count={gt_count} euros={_safe_float(gt_euros):.2f}")
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
    if team_filter:
        subdir = os.path.join(images_dir, team_filter)
        if os.path.isdir(subdir):
            paths = list(iter_image_files(subdir))
        else:
            print(f"[WARN] team_filter='{team_filter}' but folder not found: {subdir}")
            paths = []
    else:
        paths = list(iter_image_files(images_dir))
    paths = sorted(paths)
    if max_items is not None:
        paths = paths[:max_items]
    results_ok = []
    results_err = []
    missing_lines = []
    missing_ann = 0
    success_detect = 0
    fail_detect = 0
    exception_count = 0
    for p in paths:
        team = os.path.basename(os.path.dirname(p))
        fn = basename_only(p)
        key = (team, fn)
        if key not in ann_dict:
            missing_ann += 1
            missing_lines.append(f"{team}/{fn} -> annotation not found")
            continue
        gt_count = ann_dict[key].get("count", None)
        gt_euros = ann_dict[key].get("euros", None)
        if gt_count is None or gt_euros is None:
            missing_ann += 1
            missing_lines.append(
                f"{team}/{fn} -> skipped (incomplete annotation: count={gt_count}, euros={gt_euros})"
            )
            continue
        try:
            if runner is None:
                pred_count, pred_euros = run_pipeline_on_image(p, cfg)
            else:
                pred_count, pred_euros = runner(p)
            pred_count = int(pred_count)
            pred_euros = float(pred_euros)
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
    evaluated_images = len(results_ok)
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
        write_report_txt(report_path, summary_lines, [], error_lines, missing_lines)
        print(f"Report saved to: {report_path}")
        return
    count_errs = [(r["pred_count"] - r["gt_count"]) for r in results_ok]
    euro_errs = [(r["pred_euros"] - r["gt_euros"]) for r in results_ok]
    count_mae = mae(count_errs)
    count_rmse = rmse(count_errs)
    count_acc = sum(1 for e in count_errs if abs(e) <= tol_count) / len(count_errs)
    euro_mae = mae(euro_errs)
    euro_rmse = rmse(euro_errs)
    euro_acc = sum(1 for e in euro_errs if abs(e) <= tol_euro) / len(euro_errs)
    correct_count_images = sum(1 for r in results_ok if r["pred_count"] == r["gt_count"])
    correct_money_images = sum(1 for r in results_ok if abs(r["pred_euros"] - r["gt_euros"]) <= tol_euro)
    correct_both_images = sum(
        1 for r in results_ok
        if (r["pred_count"] == r["gt_count"]) and (abs(r["pred_euros"] - r["gt_euros"]) <= tol_euro)
    )
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
    error_lines = []
    for r in results_err:
        error_lines.append(
            f"{r['team']}/{r['file']} | "
            f"GT count={r['gt_count']} GT €={_safe_float_local(r['gt_euros']):.2f} | "
            f"ERROR: {r['err']}"
        )
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
    print("=================================================")
    print("EVALUATION")
    print("images_dir:", images_dir)
    print("team_filter:", team_filter)
    print(f"evaluated images (no exception): {evaluated_images}")
    print(f"missing annotations: {missing_ann} | exceptions: {exception_count}")
    print("-------------------------------------------------")
    print(f"Detection success (pred_count>0): {success_detect}/{evaluated_images}")
    print(f"Detection fail (pred_count==0): {fail_detect}/{evaluated_images}")
    print("-------------------------------------------------")
    print(f"[COUNT] MAE={count_mae:.3f} RMSE={count_rmse:.3f} Accuracy(|err|<={tol_count})={count_acc*100:.1f}%")
    print(f"[EURO ] MAE={euro_mae:.3f} RMSE={euro_rmse:.3f} Accuracy(|err|<={tol_euro:.2f}€)={euro_acc*100:.1f}%")
    print("-------------------------------------------------")
    print(f"Correct coin count images : {correct_count_images}/{evaluated_images}")
    print(f"Correct monetary value images : {correct_money_images}/{evaluated_images} (±{tol_euro:.2f}€)")
    print(f"Correct BOTH (count + value) : {correct_both_images}/{evaluated_images}")
    print("=================================================")
    write_report_txt(report_path, summary_lines, mismatch_lines, error_lines, missing_lines)
    print(f"Report saved to: {report_path}")