import cv2
import numpy as np

# 0=otsu, 1=adaptive, 2=mser, 3=kmeans_color

def seg_otsu(enhanced):
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def seg_adaptive(enhanced, block_size=11, C=2):
    return cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C
    )

def seg_mser(gray):
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    mask = np.zeros_like(gray)
    for pts in regions:
        cv2.fillPoly(mask, [pts], 255)
    binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)[1]
    return binary

def seg_kmeans_color(img_bgr, k=2):
    pixels = img_bgr.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()].reshape(img_bgr.shape)

    color_gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(color_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def seg_hybrid_adaptive_edge(enhanced, gray, canny_low=50, canny_high=150):
    adaptive = seg_adaptive(enhanced, block_size=15, C=3)  # Larger block for smoother
    edges = cv2.Canny(gray, canny_low, canny_high)
    binary = cv2.bitwise_or(adaptive, edges)
    return binary

def apply_segmentation(method_id, img_bgr, gray, enhanced):
    if method_id == 0:
        return seg_otsu(enhanced), "otsu"
    if method_id == 1:
        return seg_adaptive(enhanced), "adaptive"
    if method_id == 2:
        return seg_mser(gray), "mser"
    if method_id == 3:
        return seg_kmeans_color(img_bgr), "kmeans_color"
    if method_id == 4:
        return seg_hybrid_adaptive_edge(enhanced, gray), "hybrid_adaptive_edge"
    raise ValueError(f"Unknown SEG_METHOD_ID={method_id}. Use 0..3.")