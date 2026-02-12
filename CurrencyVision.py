import numpy as np
import cv2 as cv 

# -----------------------------
# Chargement image
# -----------------------------
path = "dataset/images/3.jpg"
img = cv.imread(path)

if img is None:
    raise FileNotFoundError("Image introuvable")

c_img = cv.resize(img, (0, 0), fx=0.4, fy=0.4)
gray = cv.cvtColor(c_img, cv.COLOR_BGR2GRAY)

# -----------------------------
# Détection des cercles
# -----------------------------
blur = cv.medianBlur(gray, 5)

circles = cv.HoughCircles(
    blur,
    cv.HOUGH_GRADIENT,
    dp=1,
    minDist=80,
    param1=50,
    param2=70,
    minRadius=30,
    maxRadius=200
)

if circles is None:
    print("Aucune pièce détectée")
    exit()

circles = np.uint16(np.around(circles))

# -----------------------------
# Rayons détectés (pixels)
# -----------------------------
r_pixels = np.array([c[2] for c in circles[0]])

# -----------------------------
# Calibration automatique
# (on suppose que la plus grande pièce = 2€)
# -----------------------------
R_2_EURO_MM = 12.88
scale = r_pixels.max() / R_2_EURO_MM  # pixels / mm
r_mm = r_pixels / scale

# -----------------------------
# Rayons officiels des pièces (mm)
# -----------------------------
coins_mm = {
    1: 8.13,     # 1 cent
    2: 9.38,     # 2 cents
    5: 10.63,    # 5 cents
    10: 9.88,    # 10 cents
    20: 11.13,   # 20 cents
    50: 12.13,   # 50 cents
    100: 11.63,  # 1 euro
    200: 12.88   # 2 euros
}

def classify_coin(r):
    return min(coins_mm, key=lambda k: abs(coins_mm[k] - r))

# -----------------------------
# Comptage + affichage
# -----------------------------
savings = 0

for (x, y, r_px), r_real in zip(circles[0], r_mm):
    value = classify_coin(r_real)
    savings += value

    # Couleur selon la valeur
    if value >= 100:
        color = (0, 0, 255)      # euros
    else:
        color = (0, 255, 0)      # cents

    cv.circle(c_img, (x, y), r_px, color, 2)
    cv.circle(c_img, (x, y), 2, color, 3)

    label = f"{value/100:.2f}€" if value >= 100 else f"{value}c"
    cv.putText(
        c_img,
        label,
        (x - 20, y),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )

# -----------------------------
# Résultat final
# -----------------------------
print(f"Total amount: {savings} cents")

cv.putText(
    c_img,
    f"Total: {savings/100:.2f} €",
    (10, c_img.shape[0] - 20),
    cv.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 0),
    2
)

cv.imshow("Coins detection", c_img)
cv.waitKey(0)
cv.destroyAllWindows()
