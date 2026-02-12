import os
import cv2
import numpy as np

# =========================================================
# Utils / 工具函数
# =========================================================
def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

def show_fit(win, image, max_w=1200, max_h=800):
    """Resize for display / 自适应缩放显示"""
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    disp = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow(win, disp)
    return disp

# =========================================================
# Path / 路径
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE_DIR, "..", "test_grp5", "18.jpg")  # 改这里

img = imread_unicode(PATH)
if img is None:
    raise FileNotFoundError(f"Cannot read image: {PATH}")

# =========================================================
# A) Color -> Gray (Point-level) / 彩色转灰度（点级处理）
# =========================================================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =========================================================
# B) Noise reduction (Local filter) / 去噪（局部滤波）
# =========================================================
blur = cv2.GaussianBlur(gray, (7, 7), 0)

# =========================================================
# C) Contrast enhancement / 对比度增强（可选）
# =========================================================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(blur)

# =========================================================
# D) Segmentation 
# =========================================================
# D1) Global Otsu 
_, otsu_binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# D2) Locally Adaptive Thresholding 
adaptive_binary = cv2.adaptiveThreshold(enhanced, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)

# D3) MSER - Maximally Stable Extremal Regions 
mser = cv2.MSER_create()
regions, _ = mser.detectRegions(gray)
mser_mask = np.zeros_like(gray)
for pts in regions:
    cv2.fillPoly(mser_mask, [pts], 255)          
mser_binary = cv2.threshold(mser_mask, 0, 255, cv2.THRESH_BINARY_INV)[1]

# D4) Color-based Segmentation (K-means in color)
pixels = img.reshape(-1, 3).astype(np.float32)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 2                                          # 
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
centers = np.uint8(centers)
segmented = centers[labels.flatten()]
color_segmented = segmented.reshape(img.shape)
color_gray = cv2.cvtColor(color_segmented, cv2.COLOR_BGR2GRAY)
_, color_binary = cv2.threshold(color_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# === Choose binary mask dùng cho phần sau  ===
binary = otsu_binary      
# =========================================================
# E) MORPHOLOGY - All suitable methods from the slide
# =========================================================
# Structuring elements
k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))   # for noise removal
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))   # for filling small holes
k_sep   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))   # for separating touching coins
k_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)) # for background estimation

# E1) Binary Dilation (expands foreground)
dilated = cv2.dilate(binary, k_sep, iterations=1)

# E2) Binary Erosion (shrinks foreground - good for separating coins)
eroded = cv2.erode(binary, k_sep, iterations=1)

# E3) Opening = Erosion → Dilation (removes small white noise)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=1)

# E4) Closing = Dilation → Erosion (fills small black holes inside coins)
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=2)

# E5) Gray-level Closing (slide 48-49) - estimates background, corrects uneven lighting
gray_closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, k_large)

# E6) Cascading Erosion + Dilation (slide 40-41) - equivalent to larger kernel, useful for separating coins
cascade = cv2.erode(binary, k_sep, iterations=2)      # strong erosion to separate
cascade = cv2.dilate(cascade, k_sep, iterations=2)    # restore size

# E7) Median Filter (rank filter, very good for salt-and-pepper noise)
median = cv2.medianBlur(closed, 5)

# E8) Majority Filter (binary rank filter - cleans binary noise)
footprint = np.ones((5, 5), np.uint8)
majority = (cv2.erode(binary, footprint, iterations=1) == 255).astype(np.uint8) * 255

# =========================================================
# Choose final mask here (only change this line)
mask = median                    


# =========================================================
# F) Morphological edge (Gradient) / 形态学边缘（梯度）
# =========================================================
k_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, k_edge)

# =========================================================
# G) Contour extraction from mask (recommended) / 从 mask 提取轮廓（推荐）
# =========================================================
# Note: better to find contours on mask (filled objects), not on edge
# 注意：轮廓提取更推荐用 mask（实心区域），edge 只是展示边缘
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

show_fit("1-Original", img)
show_fit("2-Gray", gray)
show_fit("3-Enhanced", enhanced)

show_fit("D1-Otsu Binary", otsu_binary)
# show_fit("D2-Adaptive Binary", adaptive_binary)
# show_fit("D3-MSER Binary", mser_binary)
# show_fit("D4-Color Kmeans Binary", color_binary)

# show_fit("E1-Dilated", dilated)
# show_fit("E2-Eroded", eroded)
# show_fit("E3-Opened", opened)
# show_fit("E4-Closed", closed)
# show_fit("E5-Gray Closed", gray_closed)
# show_fit("E6-Cascade", cascade)
show_fit("E7-Median", median)
# show_fit("E8-Majority", majority)

show_fit("5-Mask(clean)", mask)
show_fit("6-Edge(morph gradient)", edge)
cv2.waitKey(0)
cv2.destroyAllWindows()
