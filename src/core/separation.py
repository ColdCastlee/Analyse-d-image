import cv2
import numpy as np

def watershed_separate(img_bgr, mask, show_debug=False, show_fit=None):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)  # More iterations for better cleaning

    sure_bg = cv2.dilate(opening, kernel, iterations=4)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    
    _, sure_fg = cv2.threshold(dist_norm, 0.4, 1.0, 0)  # Higher thresh for more markers
    sure_fg = np.uint8(sure_fg * 255)
    sure_fg = cv2.erode(sure_fg, kernel, iterations=2)  # Erode more to separate
    unknown = cv2.subtract(sure_bg, sure_fg)

    if show_debug and show_fit is not None:
        show_fit("WS-sure_fg", sure_fg)
        show_fit("WS-unknown", unknown)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img_bgr, markers)

    out_mask = np.zeros_like(mask)
    out_mask[markers > 1] = 255
    return out_mask