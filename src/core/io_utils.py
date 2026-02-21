import os
import cv2
import numpy as np

def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def imwrite_unicode(path: str, img):
    """
    Write image to unicode path safely (Windows-friendly).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == "":
        ext = ".png"
        path = path + ext

    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for path={path}")
    buf.tofile(path)
    return True

def show_fit(win, image, max_w=1200, max_h=800):
    """Resize for display."""
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    disp = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(win, disp)
    return disp

def _to_bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def debug_dump(stage_name: str, img, cfg, img_path: str):
    """
    cfg:
      DEBUG_MODE: "none" | "show" | "save" | "both"
      DEBUG_OUT_DIR: output folder for saved images
    """
    mode = (cfg.get("DEBUG_MODE") or "none").lower()
    if mode == "none":
        return

    bgr = _to_bgr(img)
    if bgr is None:
        return

    base = os.path.splitext(os.path.basename(img_path))[0]
    team = os.path.basename(os.path.dirname(img_path))
    tag = f"{team}__{base}__{stage_name}"

    if mode in ("save", "both"):
        out_dir = cfg.get("DEBUG_OUT_DIR") or os.path.join("debug_out")
        out_path = os.path.join(out_dir, f"{tag}.png")
        imwrite_unicode(out_path, bgr)

    if mode in ("show", "both"):
        show_fit(stage_name, bgr)