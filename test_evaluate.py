import numpy as np
import cv2 as cv
import os
import pandas as pd

# Charger en sautant la première ligne
df = pd.read_csv("data/annotations.csv", skiprows=1)

# Renommer les colonnes
df = df.rename(columns={
    "Nom image": "image",
    "Nombre de pièces": "count",
    "Valeur monétaire €": "value",
    "Identifiant équipe": "team"
})

# Convertir les valeurs monétaires "6,18" → 6.18
df["value"] = (
    df["value"]
    .astype(str)
    .str.replace(",", ".", regex=False)
)

df["value"] = pd.to_numeric(df["value"], errors="coerce")

# Supprimer les lignes avec valeur manquante
df = df.dropna(subset=["value"])

# Garder seulement les équipes existantes
valid_teams = ["gp1", "gp2", "gp4"]
df = df[df["team"].isin(valid_teams)]

# Supprimer les images qui n'existent pas
def image_exists(row):
    path = os.path.join("data/images", row["team"], row["image"])
    return os.path.exists(path)

df = df[df.apply(image_exists, axis=1)]


def predict_image(image_path):

    original = cv.imread(image_path)

    # Sécurité si image non trouvée
    if original is None:
        return 0, 0.0

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

    contours, _ = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0, 0.0

    circles = []
    for ctr in contours:
        (x, y), radius = cv.minEnclosingCircle(ctr)
        if radius > 80:  # filtre anti-bruit
            circles.append((x, y, radius))

    if len(circles) == 0:
        return 0, 0.0

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

    coin_values = {
        "2€": 2.00,
        "1€": 1.00,
        "50c": 0.50,
        "20c": 0.20,
        "10c": 0.10,
        "5c": 0.05,
        "2c": 0.02,
        "1c": 0.01
    }

    savings = 0.0
    coin_counts = {k: 0 for k in euro_ratios.keys()}

    radii = [r for (_, _, r) in circles]
    Rmax = max(radii)

    for (_, _, r) in circles:
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

    return len(circles), round(savings, 2)


annotations = pd.read_csv("data/annotations_clean.csv")

count_errors = []
value_errors = []

correct_count = 0
correct_value = 0
correct_both = 0

for _, row in annotations.iterrows():

    image_path = os.path.join("data/images", row["team"], row["image"])

    if not os.path.exists(image_path):
        print("Image introuvable :", image_path)
        continue

    true_count = row["count"]
    true_value = row["value"]

    pred_count, pred_value = predict_image(image_path)

    count_errors.append(abs(pred_count - true_count))
    value_errors.append(abs(pred_value - true_value))

    if pred_count == true_count:
        correct_count += 1

    if abs(pred_value - true_value) <= 0.10:
        correct_value += 1

    if pred_count == true_count and abs(pred_value - true_value) <= 0.10:
        correct_both += 1

print("=== RESULTATS ===")
print("MAE count :", np.mean(count_errors))
print("MAE value :", np.mean(value_errors))
print("Accuracy count :", correct_count, "/", len(annotations))
print("Accuracy value :", correct_value, "/", len(annotations))
print("Accuracy BOTH :", correct_both, "/", len(annotations))

