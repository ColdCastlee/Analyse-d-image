import cv2
import numpy as np

# 0=dilate, 1=erode, 2=open, 3=close, 4=gray_close, 5=cascade, 6=median, 7=majority

def _kernels():
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_sep   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    return k_open, k_close, k_sep, k_large

def fill_holes(mask_bin):
    h, w = mask_bin.shape
    flood = mask_bin.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask_bin, holes)

def apply_morphology(method_id, binary, enhanced):
    k_open, k_close, k_sep, k_large = _kernels()

    if method_id == 0:      # dilate
        mask = cv2.dilate(binary, k_sep, iterations=1)

    elif method_id == 1:    # erode
        mask = cv2.erode(binary, k_sep, iterations=1)

    elif method_id == 2:    # open
        mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=1)

    elif method_id == 3:    # close
        mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=2)

    elif method_id == 4:    # gray_close -> then threshold
        gray_closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, k_large)
        _, mask = cv2.threshold(gray_closed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    elif method_id == 5:    # cascade (erode then dilate)
        mask = cv2.erode(binary, k_sep, iterations=2)
        mask = cv2.dilate(mask, k_sep, iterations=2)

    elif method_id == 6:
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=1)  
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_open, iterations=1)
        mask = cv2.medianBlur(opened, 5)

    elif method_id == 7:    # majority
        footprint = np.ones((5, 5), np.uint8)
        mask = (cv2.erode(binary, footprint, iterations=1) == 255).astype(np.uint8) * 255

    else:
        raise ValueError(f"Unknown MORPH_METHOD_ID={method_id}. Use 0..7.")

    mask = fill_holes(mask)
    return mask