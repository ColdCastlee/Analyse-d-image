# Euro Coin Detection and Value Estimation

## Project Overview

This project implements a complete computer vision pipeline for:

* detecting euro coins in images
* separating overlapping coins
* classifying coin types
* estimating total monetary value
* evaluating performance against ground truth annotations

The system is based on classical image processing techniques combined with feature-based matching.

---

## Processing Pipeline

```
Input image
→ preprocessing
→ segmentation
→ morphological post-processing
→ object separation
→ coin detection
→ coin classification and value estimation
→ performance evaluation
```

---

## Detailed Steps

### 1. Preprocessing

* grayscale conversion
* noise reduction (blur)
* optional contrast enhancement

### 2. Segmentation

* global Otsu thresholding (default)

### 3. Morphological post-processing

* closing operation
* median filtering
* mask cleaning and refinement

### 4. Object separation

* watershed algorithm for overlapping coins

### 5. Coin detection

* connected component analysis
* local circle estimation (Hough-based)

### 6. Classification and value estimation

* ORB feature matching with reference images
* RANSAC validation
* fallback to real-scale estimation using bimetal coins (1€ or 2€)

### 7. Evaluation

* comparison with ground truth annotations
* count accuracy
* monetary error
* report generation

---

## Project Structure

```
Analyse-d-image/
│
├── data/
│   ├── images/
│   │   ├── gp1/
│   │   ├── gp2/
│   │   ├── gp4/
│   │   └── gp5/
│   │
│   ├── ref/                  # reference coin images
│   ├── annotations.csv       # ground truth labels
│
├── debug_out/                # saved debug visualizations
├── outputs/                  # optional outputs
│
├── src/
│   ├── main.py
│   └── core/
│       ├── preprocess.py
│       ├── morphology.py
│       ├── detection.py
│       ├── classification.py
│       ├── evaluator.py
│       └── ...
```

---

## Configuration

Pipeline parameters are defined in:

```
src/main.py → CFG dictionary
```

### Main Parameters

| Parameter          | Description                |
| ------------------ | -------------------------- |
| SEG_METHOD_ID      | segmentation method        |
| MORPH_METHOD_ID    | mask cleaning method       |
| SEP_METHOD_ID      | object separation method   |
| DETECT_METHOD_ID   | geometric detection method |
| CLASSIFY_METHOD_ID | classification strategy    |

---

## Running Modes

The program supports two execution modes.

---

### Single Image Debug Mode

Visualizes intermediate processing stages.

Edit `src/main.py`:

```
RUN_DEBUG_SINGLE = True
DEBUG_MODE = "show" / "save" / "both"
DEBUG_IMAGE_PATH = data/images/gp4/5.jpg
```

Run:

```
python src/main.py
```

---

### Batch Evaluation Mode (Full Dataset)

Processes the entire dataset and computes performance metrics.

Edit:

```
RUN_DEBUG_SINGLE = False
```

Run:

```
python src/main.py
```

Output file:

```
evaluation_report.txt
```

Metrics include:

* coin count accuracy
* monetary value error
* MAE / RMSE
* success rate

---

## Classification Methods

| ID | Method                                     |
| -- | ------------------------------------------ |
| 0  | real-scale estimation using bimetal anchor |
| 1  | radius ratio method                        |
| 2  | ORB feature matching with reference images |

---

## Requirements

Install dependencies:

```
pip install opencv-python numpy matplotlib pandas
```

---

## Output

The system produces:

* detected coin count
* predicted coin types
* total monetary value
* evaluation report

Optional debug outputs:

* segmentation masks
* intermediate pipeline results
* detection visualization

---

## Limitations

* scale estimation fails if no bimetal coin is detected
* segmentation sensitive to lighting conditions
* overlapping coins may be over- or under-separated
* feature matching depends on reference image quality

---

## Future Improvements

* deep learning-based segmentation
* robust scale estimation without anchor coin
* improved separation of touching coins
* machine learning classification models
