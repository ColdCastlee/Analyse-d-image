import cv2
import numpy as np

def watershed_separate(img_bgr, mask, show_debug=False, show_fit=None,
                          k_open=3, open_iter=0,
                          k_close=5, close_iter=1,
                          bg_dilate_iter=2,
                          dist_thresh=0.35,
                          fg_dilate_iter=1,
                          min_fg_area=30):
    """
    Robust watershed for touching coins.
    Key ideas:
      - Prefer CLOSE (fill holes) over OPEN (which can erase faint coins)
      - Lower dist_thresh (0.30~0.45) to keep enough seeds
      - Avoid heavy erosion on sure_fg
      - Constrain output inside original mask
    """
    # ensure binary 0/255
    m = (mask > 0).astype(np.uint8) * 255

    kO = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
    kC = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))

    # 1) Fill small gaps/holes in coins (important when lighting uneven)
    m2 = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kC, iterations=close_iter)

    # 2) OPTIONAL tiny open to remove salt noise (often set open_iter=0)
    if open_iter > 0:
        m2 = cv2.morphologyEx(m2, cv2.MORPH_OPEN, kO, iterations=open_iter)

    # 3) sure background
    sure_bg = cv2.dilate(m2, kO, iterations=bg_dilate_iter)

    # 4) distance transform for sure foreground seeds
    dist = cv2.distanceTransform(m2, cv2.DIST_L2, 5)
    # normalize only for debug viewing (threshold uses raw dist)
    _, sure_fg = cv2.threshold(dist, dist_thresh * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # make seeds a bit larger (instead of eroding them)
    if fg_dilate_iter > 0:
        sure_fg = cv2.dilate(sure_fg, kO, iterations=fg_dilate_iter)

    # remove tiny seed speckles (noise)
    nlab, lab = cv2.connectedComponents(sure_fg)
    if nlab > 1:
        cleaned = np.zeros_like(sure_fg)
        for i in range(1, nlab):
            area = int((lab == i).sum())
            if area >= min_fg_area:
                cleaned[lab == i] = 255
        sure_fg = cleaned

    unknown = cv2.subtract(sure_bg, sure_fg)

    if show_debug and show_fit is not None:
        show_fit("WS2_mask_in", m)
        show_fit("WS2_mask_close", m2)
        # dist for view
        dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        show_fit("WS2_dist", dist_vis)
        show_fit("WS2_sure_fg", sure_fg)
        show_fit("WS2_unknown", unknown)

    # 5) markers
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # 6) watershed
    markers = cv2.watershed(img_bgr, markers)

    # IMPORTANT: keep only inside original mask to avoid losing valid fg
    out_mask = np.zeros_like(m)
    out_mask[(markers > 1) & (m > 0)] = 255
    return out_mask