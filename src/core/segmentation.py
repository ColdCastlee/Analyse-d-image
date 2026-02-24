import cv2
import numpy as np

# 0=otsu, 1=adaptive, 2=mser, 3=kmeans_color

def seg_otsu(enhanced):
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def seg_adaptive(enhanced, block_size=31, C=1):
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
    # Convert to grayscale for intensity-based clustering
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Light equalization to handle uneven lighting
    gray_eq = cv2.equalizeHist(gray)
    
    # Apply light blur to reduce noise
    gray_blur = cv2.medianBlur(gray_eq, 3)
    
    # Perform k-means on grayscale (1 channel)
    pixels = gray_blur.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    
    # Determine background cluster (the darker one)
    background_idx = np.argmin(centers)
    
    # Create binary mask: foreground=255, background=0
    mask = np.zeros(labels.shape[0], dtype=np.uint8)
    mask[labels.flatten() != background_idx] = 255
    binary = mask.reshape(img_bgr.shape[:2])
    
    return binary

def seg_hybrid_adaptive_edge(enhanced, gray, block_size=51, C=-2, canny_low=30, canny_high=100): 
    adaptive = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,  # Change to BINARY for bright foreground (coins)
        block_size, C
    ) 
    edges = cv2.Canny(gray, canny_low, canny_high)
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)  # Connect coin edges
    binary = cv2.bitwise_or(adaptive, edges_dilated)
    
    # Optional: Close to fill small holes in coins
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
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