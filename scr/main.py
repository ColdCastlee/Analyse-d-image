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
# ===== 强力填洞 =====
h, w = mask.shape
flood = mask.copy()
ff = np.zeros((h+2, w+2), np.uint8)

cv2.floodFill(flood, ff, (0, 0), 255)   # 填背景
holes = cv2.bitwise_not(flood)
mask = cv2.bitwise_or(mask, holes)


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


####新增
# =========================================================
# Watershed separation (split touching coins)
# =========================================================

# noise removal
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

# sure background
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# distance transform
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# sure foreground (coins centers)
_, sure_fg = cv2.threshold(dist, 0.30*dist.max(), 255, 0)  # 0.35 -> 0.30
sure_fg = np.uint8(sure_fg)

# make seeds thinner to split touching coins
sure_fg = cv2.erode(sure_fg, kernel, iterations=1)


# unknown region
unknown = cv2.subtract(sure_bg, sure_fg)

# （强烈建议先看一下 watershed 的种子长什么样）
show_fit("WS-sure_fg", sure_fg)
show_fit("WS-unknown", unknown)

# markers
num_labels, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(img, markers)

mask_ws = np.zeros_like(mask)
mask_ws[markers > 1] = 255
mask = mask_ws

# ★ watershed 后重新算 edge（否则你显示的是旧 edge）
edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, k_edge)

# ★ 最终 mask 后再找 contours（你现在已经对了）
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


#
# print("Contours count:", len(contours))
# print("Mask white ratio:", np.mean(mask == 255))
#
#
# show_fit("1-Original", img)
# show_fit("2-Gray", gray)
# show_fit("3-Enhanced", enhanced)
#
# show_fit("D1-Otsu Binary", otsu_binary)
# # show_fit("D2-Adaptive Binary", adaptive_binary)
# # show_fit("D3-MSER Binary", mser_binary)
# # show_fit("D4-Color Kmeans Binary", color_binary)
#
# # show_fit("E1-Dilated", dilated)
# # show_fit("E2-Eroded", eroded)
# # show_fit("E3-Opened", opened)
# # show_fit("E4-Closed", closed)
# # show_fit("E5-Gray Closed", gray_closed)
# # show_fit("E6-Cascade", cascade)
# show_fit("E7-Median", median)
# # show_fit("E8-Majority", majority)
#
# show_fit("5-Mask(clean)", mask)
# show_fit("6-Edge(morph gradient)", edge)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




##############################coin detect ###################################
# =========================================================
# H) Detection from final mask (CC + local Hough for merged coins)
# =========================================================
circles = []

num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
areas_all = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)

med_area = float(np.median(areas_all))
min_area = 0.50 * med_area          # remove tiny noise (e.g. 16347)
split_area_th = 1.35 * med_area     # suspect merged coins (e.g. 265k)

print("CC count:", num-1, "median area:", med_area, "min_area:", min_area, "split_th:", split_area_th)

for i in range(1, num):
    area = float(stats[i, cv2.CC_STAT_AREA])
    if area < min_area:
        continue

    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]

    # ROI with padding
    pad = 25
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(img.shape[1], x + w + pad), min(img.shape[0], y + h + pad)

    roi_gray = enhanced[y0:y1, x0:x1].copy()   # use enhanced (better edges)
    roi_mask = mask[y0:y1, x0:x1]

    # suppress background influence
    bg_val = int(np.median(roi_gray))
    roi_gray[roi_mask == 0] = bg_val

    if area > split_area_th:
        # ---- local Hough: try to find TWO circles in this ROI ----
        roi_blur = cv2.medianBlur(roi_gray, 5)

        # radius guess from "half area"
        r_guess = np.sqrt((area / 2.0) / np.pi)
        minR = int(max(10, 0.65 * r_guess))
        maxR = int(1.35 * r_guess)
        minDist = int(max(20, 1.2 * r_guess))

        c = cv2.HoughCircles(
            roi_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=minDist,
            param1=120,
            param2=22,          # smaller -> easier detection
            minRadius=minR,
            maxRadius=maxR
        )

        if c is not None:
            c = np.squeeze(c).astype(np.float32)
            if c.ndim == 1:
                c = c[None, :]

            # keep circles whose centers are inside mask (reduce false positives)
            kept = []
            for (cx, cy, r) in c:
                cx_i, cy_i = int(round(cx)), int(round(cy))
                if 0 <= cx_i < roi_mask.shape[1] and 0 <= cy_i < roi_mask.shape[0]:
                    if roi_mask[cy_i, cx_i] > 0:
                        kept.append((cx, cy, r))

            # sort by radius (bigger first) and take top2
            kept = sorted(kept, key=lambda t: -t[2])[:2]

            if len(kept) >= 2:
                for (cx, cy, r) in kept:
                    circles.append((int(x0 + cx), int(y0 + cy), float(r)))
                continue  # merged block handled as 2 coins

        # fallback: if local hough fails, treat as one coin (better than nothing)
        cx, cy = centroids[i]
        r = float(np.sqrt(area / np.pi))
        circles.append((int(cx), int(cy), r))
        continue

    # ---- normal single coin ----
    cx, cy = centroids[i]
    r = float(np.sqrt(area / np.pi))
    circles.append((int(cx), int(cy), r))

print("Detected coins:", len(circles))
if len(circles) == 0:
    exit()




def classify_material_adaptive(img_bgr, circles):
    """
    返回每个硬币材质:
    copper / gold / bimetal
    """

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    B = lab[...,2].astype(np.float32)
    H,W = B.shape

    feats = []

    for (cx,cy,r) in circles:

        yy,xx = np.ogrid[:H,:W]
        dist = np.sqrt((xx-cx)**2 + (yy-cy)**2)

        inner = dist < 0.5*r
        outer = (dist > 0.7*r) & (dist < 0.95*r)

        if inner.sum()<50 or outer.sum()<50:
            feats.append(("unknown",0,0))
            continue

        b_inner = np.mean(B[inner])
        b_outer = np.mean(B[outer])
        diff = abs(b_outer - b_inner)
        b_mean = np.mean(B[dist<0.95*r])

        feats.append((diff,b_mean))

    diffs = np.array([f[0] for f in feats if f[0]!="unknown"])
    means = np.array([f[1] for f in feats if f[0]!="unknown"])

    if len(diffs)==0:
        return ["unknown"]*len(circles)

    diff_th = np.median(diffs) + 2*np.std(diffs)
    mean_th = np.median(means)

    materials=[]
    for f in feats:

        if f[0]=="unknown":
            materials.append("unknown")
            continue

        diff,mean=f

        if diff>diff_th:
            materials.append("bimetal")
        else:
            materials.append("gold" if mean>mean_th else "copper")

    return materials



# =========================================================
# I) Calibration (anchor by bimetal) + Classification + Visualization
# =========================================================
materials = classify_material_adaptive(img, circles)

coins_diam_mm = {
    1: 16.25,
    2: 18.75,
    5: 21.25,
    10: 19.75,
    20: 22.25,
    50: 24.25,
    100: 23.25,
    200: 25.75
}

MAT_GROUPS = {
    "copper": [1, 2, 5],
    "gold": [10, 20, 50],
    "bimetal": [100, 200],
    "unknown": list(coins_diam_mm.keys())
}

# ---------------------------------------------------------
# 1€ vs 2€ 判定（看外环是不是金色）
# ---------------------------------------------------------
def bimetal_type_1e_or_2e(img_bgr, cx, cy, r_px):

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    B = lab[..., 2].astype(np.float32)

    H, W = B.shape
    cx, cy = int(cx), int(cy)
    r = float(r_px)

    yy, xx = np.ogrid[:H, :W]
    dist = np.sqrt((xx - cx)**2 + (yy - cy)**2)

    outer = (dist >= 0.72*r) & (dist <= 0.95*r)
    inner = (dist >= 0.10*r) & (dist <= 0.45*r)

    if outer.sum() < 50 or inner.sum() < 50:
        return 100

    b_outer = float(np.median(B[outer]))
    b_inner = float(np.median(B[inner]))

    return 100 if (b_outer - b_inner) > 3 else 200


# ---------------------------------------------------------
# 大字标签
# ---------------------------------------------------------
def draw_label(out, text, org, scale=1.2, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = int(org[0]), int(org[1])

    cv2.rectangle(out, (x-4, y-th-6), (x+tw+6, y+6), (0,0,0), -1)
    cv2.putText(out, text, (x, y), font, scale, (255,255,255), thickness, cv2.LINE_AA)


# ---------------------------------------------------------
# 1) 像素直径
# ---------------------------------------------------------
diam_px = np.array([2.0 * float(r) for (_,_,r) in circles], dtype=np.float32)

# ---------------------------------------------------------
# 2) 用 bimetal 做标尺
# ---------------------------------------------------------
bimetal_idxs = [i for i, m in enumerate(materials) if m == "bimetal"]

scale = None

if len(bimetal_idxs) > 0:

    # 最大那个最稳定
    bi = max(bimetal_idxs, key=lambda i: diam_px[i])

    cx, cy, r = circles[bi]
    coin_type = bimetal_type_1e_or_2e(img, cx, cy, r)

    scale = diam_px[bi] / coins_diam_mm[coin_type]

    print(f"[ANCHOR] coin index={bi}, type={coin_type}, scale(px/mm)={scale:.4f}")

else:
    raise RuntimeError("No bimetal coin detected → cannot anchor scale")

# ---------------------------------------------------------
# 3) 转 mm
# ---------------------------------------------------------
d_mm = diam_px / scale

# ---------------------------------------------------------
# 4) 在材质组内选最近直径
# ---------------------------------------------------------
values_list = []

# --- 组内排序匹配：更稳，不容易互相串 ---
values_list = [None] * len(circles)

def assign_by_ranking(idxs, allowed_values):
    """idxs: circle indices in this material group"""
    if len(idxs) == 0:
        return
    # sort circles by measured diameter
    idxs_sorted = sorted(idxs, key=lambda i: float(d_mm[i]))

    # sort official candidates by official diameter
    allowed_sorted = sorted(allowed_values, key=lambda v: coins_diam_mm[v])

    # 如果数量不一致：做一个“最近邻但带顺序”的匹配（min cost monotonic）
    # 简化：先按数量截断到 min(n, m)，剩下的再最近邻补
    n = len(idxs_sorted)
    m = len(allowed_sorted)
    k = min(n, m)

    # 先顺序对齐前 k 个
    for i in range(k):
        values_list[idxs_sorted[i]] = int(allowed_sorted[i])

    # 如果该组检测到的比候选多（很少发生），剩余用最近邻补
    if n > m:
        for j in range(m, n):
            ii = idxs_sorted[j]
            values_list[ii] = int(min(allowed_sorted, key=lambda v: abs(coins_diam_mm[v] - float(d_mm[ii]))))

# --- 分组 index ---
idx_copper  = [i for i, mat in enumerate(materials) if mat == "copper"]
idx_gold    = [i for i, mat in enumerate(materials) if mat == "gold"]
idx_bimetal = [i for i, mat in enumerate(materials) if mat == "bimetal"]
idx_unknown = [i for i, mat in enumerate(materials) if mat not in ("copper", "gold", "bimetal")]

assign_by_ranking(idx_copper,  MAT_GROUPS["copper"])    # 1/2/5
assign_by_ranking(idx_gold,    MAT_GROUPS["gold"])      # 10/20/50

# bimetal 只有 1€ / 2€，用你原来的外环判断更稳（不排序）
for i in idx_bimetal:
    cx, cy, r = circles[i]
    values_list[i] = int(bimetal_type_1e_or_2e(img, cx, cy, r))

# unknown 的再用全体最近邻兜底
for i in idx_unknown:
    v = min(coins_diam_mm.keys(), key=lambda k: abs(coins_diam_mm[k] - float(d_mm[i])))
    values_list[i] = int(v)

# safety
values_list = [int(v) for v in values_list]


print("Materials:", materials)
print("Diameters(mm):", [round(float(x),2) for x in d_mm])
print("Pred values:", values_list)

# ---------------------------------------------------------
# 5) 画图 + 统计
# ---------------------------------------------------------
out = img.copy()
total_cents = 0

for (cx,cy,r_px), v, mat in zip(circles, values_list, materials):

    total_cents += int(v)

    if mat == "copper":
        color = (0,140,255)
    elif mat == "gold":
        color = (0,255,255)
    elif mat == "bimetal":
        color = (255,255,0)
    else:
        color = (0,255,0)

    cv2.circle(out, (int(cx),int(cy)), int(round(r_px)), color, 2)
    cv2.circle(out, (int(cx),int(cy)), 2, color, 3)

    label = f"{v/100:.2f}€" if v>=100 else f"{v}c"

    draw_label(out, label, (int(cx)-25, int(cy)-int(r_px)-10))

draw_label(out, f"Total: {total_cents/100:.2f} euro",
           (10, out.shape[0]-15), scale=1.3)

print("Detected coins:", len(circles))
print("Total amount:", total_cents, "cents")

show_fit("FINAL RESULT", out)
cv2.waitKey(0)
cv2.destroyAllWindows()

