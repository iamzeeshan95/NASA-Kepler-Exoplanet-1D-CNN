# NASA Kepler Exoplanet Transit Detection using 1D CNN

An end-to-end deep learning pipeline to detect exoplanet transits from NASA Kepler stellar light-curve time-series data. This project benchmarks a **1D Convolutional Neural Network (1D CNN)** against a domain-engineered **HistGradientBoosting** baseline, featuring a leak-free processing pipeline designed for extreme class imbalance.



##  Overview

When an exoplanet passes in front of its host star relative to an observer, it causes a temporary dip in the star's measured brightness (a **transit**). Identifying these tiny, periodic signals within thousands of time-series points per star is a major data science challenge due to noise, stellar variability, and look-alikes like eclipsing binary stars.

This repository implements:
1. **Domain Feature Engineering:** Extracted physical light-curve properties (dip depth, skewness, flux variance, noise floor, and odd-vs-even segment depth asymmetry).
2. **Tabular Baseline Model:** Gradient Boosted Decision Trees trained on domain features with class-weighted loss.
3. **1D Convolutional Neural Network:** Deep time-series classifier processing normalized flux vectors to detect subtle transit signals.
4. **Leak-Free Processing & Validation:** Robust handling of severe class imbalance (~1:130 positive-to-negative ratio) without synthetic data leakage.

---

##  Dataset & Physics Background

The dataset consists of stellar light-curves from the **NASA Kepler Space Telescope** (`exoTrain.csv` and `exoTest.csv`):
* **Training Set:** 5,087 stars $\times$ 3,197 time steps (~37 confirmed exoplanets)
* **Test Set:** 570 stars $\times$ 3,197 time steps (5 confirmed exoplanets)
* **Target Labels:** `0` = Non-Exoplanet / False Positive | `1` = Confirmed Exoplanet

---

##  Key Data Science Pitfalls & Engineering Fixes

### Problem 1: Naive SMOTE Oversampling & Data Leakage
* **The Issue:** Initially applying global SMOTE oversampling to high-dimensional raw time-series data creates synthetic noise and causes severe over-fitting. Furthermore, splitting validation sets *after* oversampling leaks near-identical synthetic samples into the validation split, resulting in artificial `val_pr_auc = 1.0000` while failing on real test data (`100% false positive rate`).
* **The Fix:** We established a strict, leak-free pipeline:
  1. Carved out a clean validation split **before** applying any oversampling.
  2. Applied mild, constrained SMOTE (`sampling_strategy=0.25`) **only** to the training sub-split.
  3. Scaled tabular features with `StandardScaler` fitted strictly on training data.

### Problem 2: Misleading Accuracy in Extreme Imbalance
* **The Issue:** With >99% negative samples, a dummy model predicting all stars as `0` achieves >99% accuracy while finding zero exoplanets. Standard 0.5 probability thresholds often cause model predictions to collapse to a single class.
* **The Fix:** We prioritized **Precision-Recall Area Under the Curve (PR-AUC / Average Precision)** over accuracy, and dynamically tuned the prediction decision threshold on the clean validation set by maximizing the $F_1$ score.

---

##  Pipeline Workflow

```
┌──────────────────────────────────────────────────────────┐
│              Raw Light-Curve CSV Data                    │
└────────────────────────────┬─────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
┌───────────────────────┐         ┌───────────────────────┐
│ Row-Wise Detrending   │         │ Domain Feature        │
│ & Normalization       │         │ Extraction            │
└───────────┬───────────┘         └───────────┬───────────┘
            │                                 │
            ▼                                 ▼
┌───────────────────────┐         ┌───────────────────────┐
│ Clean Val Split       │         │ Fit StandardScaler    │
│ + Mild SMOTE (Train)  │         │ (Train Set Only)      │
└───────────┬───────────┘         └───────────┬───────────┘
            │                                 │
            ▼                                 ▼
┌───────────────────────┐         ┌───────────────────────┐
│ 1D CNN Architecture   │         │ HistGradientBoosting  │
│ (Time-Series)         │         │ (Tabular Baseline)    │
└───────────┬───────────┘         └───────────┬───────────┘
            │                                 │
            └────────────────┬────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ Val Threshold Tuning (F1-Optimal) & PR-AUC Evaluation    │
└──────────────────────────────────────────────────────────┘
```

---

##  Model Architecture (1D CNN)

```python
model = Sequential([
    Input(shape=(3197, 1)),
    
    Conv1D(16, kernel_size=15, strides=2, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=4),

    Conv1D(32, kernel_size=11, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=4),

    Conv1D(64, kernel_size=7, activation='relu', padding='same'),
    BatchNormalization(),
    GlobalAveragePooling1D(),

    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

---

##  Results & Benchmarks

Evaluation performed on the independent 570-star test set (5 positive, 565 negative):

| Metric | Tabular Baseline (HistGB) | 1D CNN Model |
| :--- | :---: | :---: |
| **PR-AUC (Average Precision)** | 0.1222 | **0.9670** |
| **Recall (Exoplanets Detected)** | 4 / 5 (80%) | **5 / 5 (100%)** |
| **Precision** | 0.8000 | **0.8333** |
| **False Positives** | 1 | **1** |
| **False Negatives** | 1 | **0** |

### Test Confusion Matrix (1D CNN)
```
                  Predicted Non-Planet (0)   Predicted Exoplanet (1)
Actual Non-Planet          564                          1
Actual Exoplanet             0                          5
```

---
### Note on Run-to-Run Variance & Threshold Tuning:
Across local CPU and Colab T4 runs on the 5-exoplanet test split, results vary slightly due to random weight initialization, validation $F_1$ threshold selection, and CPU vs. GPU floating-point execution. Tuning the threshold for maximum $F_1$ can push precision to 100% (0 false positives) while trading off a couple of borderline or unusual transits.

##  Error Analysis

Inspecting the test predictions revealed critical insights:
* **True Positives (5/5):** All 5 exoplanets in the test set were successfully flagged with high output confidence ($p > 0.94$).
* **False Positive Case (1 star):** The single misclassified non-planet displayed a steep negative brightness dip (`min_dip ≈ -5.50`), high stellar noise (`flux_std ≈ 19,455`), and an asymmetrical dip profile (`odd_even_diff ≈ 1.48`). These characteristics mimic an eclipsing binary system or starspot activity, illustrating the main source of false positives in astronomical surveys.

### Threshold & Precision–Recall Trade-Off:
On a test set with only 5 confirmed exoplanets, selecting a high $F_1$-optimal threshold on the validation split can eliminate false positives (0 FP) at the cost of missing shallow or unusually asymmetric transits (2 FN)—highlighting why inspecting both PR-AUC and error cases matters more than relying on a single run.

---

##  Installation & Quick Start

### Prerequisites
* Python 3.10+
* Jupyter Notebook / VS Code

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/iamzeeshan95/NASA-Kepler-Exoplanet-1D-CNN.git

cd nasa-kepler-1d-cnn

pip install numpy pandas matplotlib scipy scikit-learn imbalanced-learn tensorflow
```

### 2. Dataset Setup
Download the Exoplanet Hunting in Deep Space dataset from Kaggle: https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data which contains the `exoTrain.csv` and `exoTest.csv`, and place them directly in the root directory.

### 3. Run the Notebook
Launch VS Code or Jupyter Notebook and run `kepler1D.ipynb` from start to finish:
```bash
jupyter notebook kepler1D.ipynb
```
---
##### Made with ❤ by Muhammad Zeeshan
---