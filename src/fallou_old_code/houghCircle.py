import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt

# --- Chargement ---
original = cv.imread("dataset/images/8.jpg")
detected = original.copy()

# --- Prétraitement ---
gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)

# Blur fort pour lisser les textures internes
blur = cv.GaussianBlur(gray, (9, 9), 2)

# --- Détection par HoughCircles ---
circles = cv.HoughCircles( 
    blur,
    cv.HOUGH_GRADIENT,
    dp=1.2,
    minDist=150,
    param1=100,
    param2=50,
    minRadius=60,
    maxRadius=200
)

if circles is None:
    print("Aucune pièce détectée")
    exit()

circles = np.uint16(np.around(circles[0]))

# --- Ratios réels des pièces euro ---
euro_ratios = {
    "2e": 1.00,
    "50c": 0.94,
    "1e": 0.90,
    "20c": 0.86,
    "5c": 0.82,
    "10c": 0.77,
    "2c": 0.73,
    "1c": 0.63
}
# valeur des pièces
coin_values = {
    "2e": 2.00,
    "1e": 1.00,
    "50c": 0.50,
    "20c": 0.20,
    "10c": 0.10,
    "5c": 0.05,
    "2c": 0.02,
    "1c": 0.01
}

# --- Normalisation par plus grand rayon ---
radii = [r for (_, _, r) in circles]
Rmax = max(radii)

coin_counts = {k: 0 for k in euro_ratios.keys()}
classified = []

savings = 0
# --- Classification ---
for (x, y, r) in circles:
    ratio = r / Rmax

    best_label = None
    best_diff = float("inf")

    for coin, ref in euro_ratios.items():
        diff = abs(ratio - ref)
        if diff < best_diff:
            best_diff = diff
            best_label = coin

    coin_counts[best_label] += 1
    savings += coin_values[best_label] 
    classified.append((x, y, r, best_label))

# --- Dessin ---
for (x, y, r, label) in classified:
    cv.circle(detected, (x, y), r, (0, 255, 0), 3)

    text = f"{label}\nr={r}px"

    # Ligne 1
    cv.putText(detected,
               label,
               (x - 30, y - 40),
               cv.FONT_HERSHEY_SIMPLEX,
               0.7,
               (0, 0, 255),
               2)

    # Ligne 2
cv.putText(detected,
        f"Total = {savings:.2f} Euro",
        (30, 50),
        cv.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 0, 0),
        3)

# --- Affichage ---
plt.figure(figsize=(8, 8))
plt.title("Pièces détectées : " + ", ".join([f"{k}: {v}" for k, v in coin_counts.items() if v > 0]))
plt.imshow(cv.cvtColor(detected, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
