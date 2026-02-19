import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt

original = cv.imread("C:/Users/hp/Desktop/AnalyseImage/Currency_Vision/dataset/images/3.jpg")
detected = original.copy()
img = original.copy()


gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (9, 9), 0)

thresh = cv.adaptiveThreshold(
    blur, 255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY_INV,
    31, 5
)


kernel = np.ones((3,3), np.uint8)
opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

# --- Détection des contours ---
contours, hierarchy = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

if len(contours) == 0:
    print("Aucune pièce détectée")
    exit()

# --- Extraction des cercles ---
circles = []
for ctr in contours:
    (x, y), radius = cv.minEnclosingCircle(ctr)
    if radius > 60:  # filtre anti-bruit
        circles.append((x, y, radius))

if len(circles) == 0:
    print("Aucune pièce détectée")
    exit()

euro_ratios = {
    "2€": 1.00,
    "50c": 0.94,
    "1€": 0.90,
    "20c": 0.86,
    "5c": 0.82,
    "10c": 0.77,
    "2c": 0.73,
    "1c": 0.63
}

# --- Classification par ratio ---
radius = [r for (_, _, r) in circles]
Rmax = max(radius)
Rmin = min(radius)
print(Rmax)
print(Rmin)
# --- Comptage ---
coin_counts = {k: 0 for k in euro_ratios.keys()}
classified = []

# --- Classification par ratio ---
for (x, y, r) in circles:
    ratio = r / Rmax

    best_label = None
    best_diff = 999

    for coin, ref in euro_ratios.items():
        diff = abs(ratio - ref)
        if diff < best_diff:
            best_diff = diff
            best_label = coin

    coin_counts[best_label] += 1
    classified.append((x, y, r, best_label))

# --- Dessin ---
for (x, y, r, label) in classified:
    cv.circle(detected, (int(x), int(y)), int(r), (0, 255, 0), 3)
    cv.putText(detected, label, (int(x - 20), int(y - 20)),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# --- Affichage ---
plt.figure(figsize=(10,10))
plt.title("Pièces détectées : " + ", ".join([f"{k}: {v}" for k,v in coin_counts.items()]))
plt.imshow(cv.cvtColor(detected, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
