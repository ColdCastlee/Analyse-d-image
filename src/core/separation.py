import cv2
import numpy as np

def watershed_separate(img_bgr, mask, show_debug=False, show_fit=None):
    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.30 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_fg = cv2.erode(sure_fg, kernel, iterations=1)
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