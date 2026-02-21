import cv2
import numpy as np

def imread_unicode(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def show_fit(win, image, max_w=1200, max_h=800):
    """Resize for display."""
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    disp = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(win, disp)
    return disp