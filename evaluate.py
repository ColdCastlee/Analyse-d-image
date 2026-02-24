
import os
import pandas as pd
import numpy as np
from test import predict_image
import cv2 as cv

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

def predict_image(image_path):

    original = cv.imread(image_path)
    gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 2)

    circles = cv.HoughCircles(
        blur,
        cv.HOUGH_GRADIENT,
        dp=1.2,
        minDist=150,
        param1=100,
        param2=70,
        minRadius=80,
        maxRadius=200
    )

    if circles is None:
        return 0, 0

    circles = np.uint16(np.around(circles[0]))

    radii = [r for (_, _, r) in circles]
    Rmax = max(radii)

    savings = 0

    for (_, _, r) in circles:
        ratio = r / Rmax

        best_label = None
        best_diff = float("inf")

        for coin, ref in euro_ratios.items():
            diff = abs(ratio - ref)
            if diff < best_diff:
                best_diff = diff
                best_label = coin

        savings += coin_values[best_label]

    return len(circles), savings


# ==============================
# 1️⃣ Charger et nettoyer le CSV
# ==============================

annotations = pd.read_csv("data/annotations.csv", header=1)

annotations.columns = ["image", "nb_pieces", "value", "team"]

# Nettoyage colonne value (virgule -> point)
annotations["value"] = (
    annotations["value"]
    .astype(str)
    .str.replace(",", ".", regex=False)
)

annotations["value"] = pd.to_numeric(annotations["value"], errors="coerce")

# Supprimer lignes avec valeurs manquantes
annotations = annotations.dropna()

print("Dataset size:", len(annotations))
print(annotations.dtypes)
print()

# ==============================
# 2️⃣ Initialisation métriques
# ==============================

count_errors = []
value_errors = []

correct_count = 0
correct_value = 0
correct_both = 0

total_images = 0

# ==============================
# 3️⃣ Boucle d’évaluation
# ==============================

for _, row in annotations.iterrows():
    
    image_path = os.path.join("data/images", row["team"], row["image"])

    # Vérifier que l’image existe
    if not os.path.exists(image_path):
        print("Image manquante :", image_path)
        continue

    true_count = row["nb_pieces"]
    true_value = row["value"]

    try:
        pred_count, pred_value = predict_image(image_path)
    except Exception as e:
        print("Erreur sur image :", image_path)
        print(e)
        continue

    total_images += 1

    # Erreurs
    count_errors.append(abs(pred_count - true_count))
    value_errors.append(abs(pred_value - true_value))

    # Accuracy
    if pred_count == true_count:
        correct_count += 1

    if abs(pred_value - true_value) <= 0.10:
        correct_value += 1

    if pred_count == true_count and abs(pred_value - true_value) <= 0.10:
        correct_both += 1


# ==============================
# 4️⃣ Calcul des métriques
# ==============================

if total_images == 0:
    print("Aucune image valide trouvée.")
else:
    MAE_count = np.mean(count_errors)
    RMSE_count = np.sqrt(np.mean(np.square(count_errors)))

    MAE_value = np.mean(value_errors)
    RMSE_value = np.sqrt(np.mean(np.square(value_errors)))

    print("\n=== RESULTATS ===")
    print(f"Images évaluées: {total_images}")
    print(f"[COUNT] MAE={MAE_count:.3f}  RMSE={RMSE_count:.3f}")
    print(f"[EURO ] MAE={MAE_value:.3f}  RMSE={RMSE_value:.3f}")
    print(f"Accuracy count: {correct_count}/{total_images}")
    print(f"Accuracy value (±0.10€): {correct_value}/{total_images}")
    print(f"Accuracy BOTH: {correct_both}/{total_images}")