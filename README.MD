
---

```md
# Détection de pièces en euros et estimation de valeur

## Aperçu

Ce projet implémente un pipeline **100% vision classique (OpenCV)** pour :

- **détecter** les pièces en euros (cercles)
- **estimer la valeur totale** (en €) à partir des diamètres + indices couleur / matière
- **évaluer** les résultats à l’aide d’un fichier d’annotations (`annotations.csv`)

> ⚠️ Remarque : dans la version actuelle, le pipeline est volontairement **simplifié**  
> → **pas de segmentation**, **pas de morphologie**, **pas de watershed**.

---

## Pipeline actuel (version code)

```

Image (BGR)
→ Pré-traitement (gris + flou gaussien + CLAHE)
→ Détection des cercles (HoughCircles “pure”)
→ Classification des pièces (géométrie + matière + estimation d’échelle)
→ Somme des centimes → total en euros
→ Évaluation (MAE / RMSE / accuracy avec tolérances)

```

---

## Détails des étapes

### 1. Pré-traitement (`core/evaluator.py`)

- conversion en niveau de gris
- flou gaussien
- amélioration du contraste par **CLAHE**

---

### 2. Détection des pièces (`core/evaluator.py`)

- détection par **`cv2.HoughCircles`**
- **sans masque**
- **sans composantes connexes**
- **sans distance transform**

➡️ Cette approche est volontairement simple mais **très sensible au bruit**, ce qui génère **de nombreux faux positifs (pièces fantômes)**.

---

### 3. Classification / estimation de valeur (`core/classification.py`)

- normalisation photométrique (white balance simple + CLAHE sur L + filtre bilatéral)
- estimation de matière (**copper / gold / bimetal / unknown**) via :
  - statistiques robustes en espace **LAB**
  - gestion des reflets
  - score d’arête annulaire pour le bimétal
- estimation de l’échelle **px/mm**
  - priorité aux pièces bimétalliques
  - sinon estimation robuste globale
- attribution de la dénomination par **diamètre en mm** avec contrainte de matière souple
- **ORB est présent dans le code**, mais **désactivé dans la configuration actuelle car il marche pas bien**
  (`ref_db = None`)

---

## Performances (résultats actuels)

### Configuration d’évaluation

- `TOL_COUNT = 0`
- `TOL_EURO  = 0.10 €`

### Résumé global

```

Images évaluées (sans exception) : 103
Annotations manquantes           : 3
Exceptions                       : 0

Détection non vide (pred_count>0): 103 / 103

```

### Résultats quantitatifs

#### Erreur de comptage

- **MAE**  = **5.340**
- **RMSE** = **11.865**
- **Accuracy (|err| = 0)** = **49.5%**

#### Erreur monétaire

- **MAE**  = **6.453 €**
- **RMSE** = **15.221 €**
- **Accuracy (|err| ≤ 0.10 €)** = **8.7%**

#### Cas exactement corrects

- Bon nombre de pièces : **51 / 103 (49.5%)**
- Bonne valeur (±0.10 €) : **9 / 103 (8.7%)**
- Bon nombre **et** bonne valeur : **7 / 103 (6.8%)**

---

## Analyse des erreurs (important)

- Les **MAE et RMSE élevés** proviennent **principalement de la sur-détection** :
  - le détecteur Hough génère **beaucoup de cercles parasites**
  - ces **pièces fantômes** augmentent artificiellement :
    - le nombre de pièces détectées
    - la valeur totale estimée
- Même lorsque les **vraies pièces sont bien détectées**, la présence de faux positifs :
  - dégrade fortement le comptage
  - amplifie l’erreur monétaire (effet cumulatif)

➡️ Le pipeline est donc **rappel-orienté (recall élevé)** mais avec une **précision faible**.

---

## Structure du projet (version actuelle)

```

Analyse-d-image/
│
├── data/
│   ├── images/               # dossiers gpX
│   └── annotations.csv       # vérité terrain
│
├── src/
│   ├── main.py
│   └── core/
│       ├── evaluator.py
│       ├── classification.py
│       ├── io_utils.py
│       └── data_loader.py

````

---

## Exécution

### Évaluation sur tout le dataset

Dans `src/main.py` :

- `RUN_DEBUG_SINGLE = False`

Puis :

```bash
python src/main.py
````

Sortie :

* `evaluation_report.txt`

---

### Debug sur une seule image

Dans `src/main.py` :

* `RUN_DEBUG_SINGLE = True`
* `DEBUG_MODE = "show" | "save" | "both" | "none"`
* `DEBUG_IMAGE_PATH = ...`

Puis :

```bash
python src/main.py
```

---

## Configuration actuelle

Dans `src/main.py` :

* `DETECT_METHOD_ID = 2`  (Hough “pure”)

---

## Dépendances

```bash
pip install opencv-python numpy
```

---

## Limitations (version actuelle)

* **Sur-détection importante** : Hough génère de nombreux cercles non pertinents.
* **Pièces très proches / collées** : détection instable aux bords.
* **Sensibilité à l’éclairage extrême** malgré la normalisation.
* **Précision monétaire faible** due à l’accumulation d’erreurs de comptage.
* **Pas de séparation d’objets** :

  * pas de segmentation
  * pas de watershed
  * chevauchements difficiles à gérer

---

## Pistes d’amélioration

* Réduction des faux positifs (filtrage géométrique / bordures / score circulaire)
* Réintroduction d’une **segmentation légère** ou d’un **watershed avec marqueurs**
* Sélection de **pièces de référence fiables**, puis estimation relative
* Réactivation contrôlée du **matching ORB** pour la validation finale

```

