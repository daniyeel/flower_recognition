# 🌸 Flower Recognition

A computer vision pipeline for classifying 7 flower species from smartphone photos using classical image processing techniques and a K-Nearest Neighbours (KNN) classifier. Achieves ~80% accuracy on a self-photographed dataset.

> **Authors:** Daniel & Joep

---

## Overview

This project takes raw flower photos, extracts a set of hand-crafted visual features via OpenCV, and feeds them into a KNN classifier. No deep learning, just image processing fundamentals: colour statistics, texture (LBP), contour analysis, and shape detection.

### Supported Species

| Label | Species |
|-------|---------|
| 0 | Dahlia 'Ellen Houston' |
| 1 | Dahlia 'Harlequin' |
| 2 | Dahlia 'Izarra' |
| 3 | Dahlia 'Wink' |
| 4 | Zinnia |
| 5 | Rudbeckia Fulgida |
| 6 | Dahlia 'Bishop of York' |

---

## How It Works

### Processing Pipeline

Each image goes through the following stages before feature extraction:

```
Original image
    ↓  Resize (÷5)
    ↓  Normalize
    ↓  Median blur
    ↓  Bilateral filter (×20)
    ↓  Background filter (remove blue/green)
    ↓  Find largest contour
    ↓  Scale contour back to original resolution
    ↓  Crop flower region
    ↓  Feature extraction
    ↓  KNN classification
```

### Feature Vector (7 dimensions)

| # | Feature | Method |
|---|---------|--------|
| 0 | Mean hue (H) | HSV colour space |
| 1 | Mean saturation (S) | HSV colour space |
| 2 | Mean value (V) | HSV colour space |
| 3 | LBP feature count | `skimage.feature.local_binary_pattern` |
| 4 | LBP unique feature count | `skimage.feature.local_binary_pattern` |
| 5 | Petal/leaf count | Contour area filtering |
| 6 | Shape vertex count | `cv2.approxPolyDP` |

---

## Dataset

All images were photographed by the authors using an iPhone 13. Each species has a dedicated `train/` and `test/` subdirectory.

**Dataset requirements:**
- At least 6 images per species in the test set
- One flower per image (the classifier focuses on the largest contour)
- Flowers must not overlap — overlapping blooms interfere with contour detection

**Expected directory structure:**

```
BloemenHerkennenFotos/
├── Dahlia ellen houston/
│   ├── train/
│   └── test/
├── Dahlia Harlequin/
│   ├── train/
│   └── test/
├── Dahlia Izarra/
│   ├── train/
│   └── test/
├── Dahlia Wink/
│   ├── train/
│   └── test/
├── Zinnia/
│   ├── train/
│   └── test/
├── Rudbeckia Fulgida/
│   ├── train/
│   └── test/
├── Dahlia Bishop of York/
│   ├── train/
│   └── test/
├── Nieuwe foto/          ← optional: single new image for live testing
└── Dump/                 ← debug output written here
```

---

## Setup

### Dependencies

```bash
pip install opencv-python numpy matplotlib scikit-image
```

### Configuration

Edit the constants at the top of `src.py` before running:

```python
directory = 'path/to/BloemenHerkennenFotos'  # Root dataset path

SCALE          = 5   # Downscale factor for processing
BLUR           = 3   # Median blur kernel size
CR_SCALE       = 2   # Crop resize scale
PARAMETER_COUNT = 7  # Feature vector length (don't change unless adding features)
DEADZONE_BGR_FILTER = 12  # Tolerance for background colour filter
```

To classify a new single image instead of running the full test set, set:

```python
TEST     = True   # Use the 'Nieuwe foto' folder as test input
TESTFOTO = 0      # Expected label for that image (for accuracy reporting)
```

### Running

```bash
python src.py
```

The script will print progress through the training and test image processing steps, then output the classification results and overall accuracy percentage.

**Approximate runtimes:**
- Processing training images: ~2 minutes
- Processing test images: ~40 seconds

Debug images for each processing stage are saved to `Dump/`.

---

## Results

~80% classification accuracy on the self-photographed test set using a 1-NN classifier.

---

## Limitations

- Paths are currently hardcoded — requires manual update of the `directory` variable
- The background filter is tuned for the specific lighting conditions of the original dataset and may need adjustment for other photos
- Only the largest contour per image is classified; images containing multiple flowers will be partially ignored
- Dataset is not publicly available (images are personal photographs)
