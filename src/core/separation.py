import cv2
import numpy as np

def watershed_separate(img_bgr, mask, show_debug=False, show_fit=None):
    kernel = np.ones((5, 5), np.uint8)

    # Close to fill small holes from shadows/noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=1)  # Reduced to 1 to prevent over-expansion

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # Increased threshold to 0.6 for larger sure_fg
    _, sure_fg = cv2.threshold(dist, 0.6 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_bg, sure_fg)

    if show_debug and show_fit is not None:
        show_fit("WS-sure_fg", sure_fg)
        show_fit("WS-unknown", unknown)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers = markers.astype(np.int32)  # Ensure int32 for watershed
    markers[unknown == 255] = 0

    # Blur image for watershed to smooth internal shadows/gradients
    img_blur = cv2.medianBlur(img_bgr, 5)
    markers = cv2.watershed(img_blur, markers)

    out_mask = np.zeros_like(mask)
    out_mask[markers > 1] = 255
    
    # Dilate to compensate for any shrinking
    out_mask = cv2.dilate(out_mask, kernel, iterations=2)
    
    # Final close to seal artifacts (optional, but keeps smooth)
    out_mask = cv2.morphologyEx(out_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return out_mask