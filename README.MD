#  Système de Détection et d'Estimation de Valeur de Pièces

## 📌 Présentation du projet

Ce projet implémente une chaîne complète de **vision par ordinateur**
permettant :

-    la détection automatique de pièces dans une image\
-    la classification de leur matériau (cuivre, or, bimétallique)\
-    l'estimation de leur valeur monétaire totale\
-    l'évaluation quantitative sur un jeu de données annoté

Le système constitue un **framework d'évaluation complet** avec
génération automatique de rapports et métriques de performance.

------------------------------------------------------------------------

## ⚙️ Pipeline de traitement

Pour chaque image :

1.  Lecture de l'image (gestion Unicode compatible Windows)
2.  Conversion en niveaux de gris
3.  Réduction du bruit (filtrage gaussien)
4.  Amélioration du contraste (CLAHE)
5.  Détection des cercles (Transformée de Hough)
6.  Classification du matériau (analyse colorimétrique LAB)
7.  Estimation de la dénomination de chaque pièce
8.  Calcul de la valeur totale
9.  Comparaison avec la vérité terrain

------------------------------------------------------------------------

##  Méthodes d'estimation des valeurs

Le système propose trois stratégies différentes.

### 🔹 Méthode 0 --- Baseline métrique basée sur le diamètre réel

-   Estimation du facteur d'échelle pixel → millimètre
-   Comparaison avec les diamètres officiels des pièces
-   Contraintes souples selon le matériau détecté

✔ Robuste\
✔ Complètement automatique\
✔ Ne nécessite pas d'images de référence

------------------------------------------------------------------------

### 🔹 Méthode 1 --- Estimation par ratio de rayon

-   Normalisation par la plus grande pièce détectée
-   Comparaison avec des ratios relatifs connus

✔ Simple et rapide\
⚠ Sensible aux erreurs de détection

------------------------------------------------------------------------

### 🔹 Méthode 2 --- Matching ORB avec vérification géométrique (CM06)

-   Filtrage des candidats par aire
-   Alignement d'échelle
-   Extraction de descripteurs ORB
-   Vérification géométrique par RANSAC
-   Fallback automatique vers la baseline si le matching est incertain

✔ Haute précision si les références sont fiables\
✔ Robuste grâce au fallback

------------------------------------------------------------------------

##  Structure du projet

    src/
    │
    ├── main.py
    │
    ├── core/
    │   ├── data_loader.py
    │   ├── evaluator.py
    │   ├── classification.py
    │   ├── matching.py
    │   └── io_utils.py
    │
    data/
    ├── images/
    ├── ref/
    └── annotations.csv

------------------------------------------------------------------------

##  Configuration

Les paramètres principaux sont définis dans `src/main.py`.

### Exemple :

``` python
CFG = {
    "DETECT_METHOD_ID": 2,
    "CLASSIFY_METHOD_ID": 0,
    "RUN_DEBUG_SINGLE": False,
    "DEBUG_MODE": "none",
    "DEBUG_OUT_DIR": "debug_out",
    "DEBUG_IMAGE_PATH": ".../gp1/25.png"
}
```

### Modes Debug

  Mode   Description
  ------ -------------------------------------
  none   Aucun affichage ni sauvegarde
  show   Affichage des étapes intermédiaires
  save   Sauvegarde des images debug
  both   Affichage + sauvegarde

------------------------------------------------------------------------

## ▶ Exécution

### Évaluation du dataset complet

``` bash
python src/main.py
```

Sorties : - Résumé dans la console - Rapport détaillé :
evaluation_report.txt

------------------------------------------------------------------------

### Débogage d'une image unique

Dans `main.py` :

``` python
RUN_DEBUG_SINGLE = True
DEBUG_MODE = "both"
```

------------------------------------------------------------------------

##  Métriques d'évaluation

Le système calcule automatiquement :

-   MAE
-   RMSE
-   Précision sous tolérance
-   Taux de détection
-   Cas d'erreurs détaillés
-   Images sans annotation
-   Exceptions

Résultats exportés dans :

    evaluation_report.txt

------------------------------------------------------------------------

##  Dépendances

Python ≥ 3.9

    pip install -r requirements.txt

------------------------------------------------------------------------

##  Auteur

Projet développé dans le cadre d'expérimentations en vision par
ordinateur.

