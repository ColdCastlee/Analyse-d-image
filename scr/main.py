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
# D) Thresholding (Global Otsu) / 阈值分割（全局 Otsu）
# =========================================================
# INV: make coins white (foreground=1) / 反阈值：让硬币为白色
_, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# =========================================================
# E) Morphology cleaning / 形态学清理（开 + 闭）
# =========================================================
k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k_open,  iterations=1)  # remove small white noise / 去小噪点
mask = cv2.morphologyEx(mask,   cv2.MORPH_CLOSE, k_close, iterations=2)  # fill small holes / 填洞
mask = cv2.medianBlur(mask, 5)                                           # stabilize boundaries / 稳定边界

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
show_fit("4-Binary(Otsu)", binary)
show_fit("5-Mask(clean)", mask)
show_fit("6-Edge(morph gradient)", edge)

cv2.waitKey(0)
cv2.destroyAllWindows()
