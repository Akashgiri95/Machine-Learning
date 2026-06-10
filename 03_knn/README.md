# 03 — K-Nearest Neighbors

A non-parametric, instance-based classifier — the simplest possible baseline before moving to model-based learners.

---

## `knn_fundamentals.ipynb`

- Five worked examples building intuition for `KNeighborsClassifier`: prediction, predicted probabilities, finding nearest neighbors (`kneighbors`), and decision boundaries
- Constructing a labelled dataset from categorical values to see how KNN handles encoded categorical features

## `knn_wine_classification.ipynb`

Multi-class classification on **scikit-learn's Wine dataset** (`load_wine`, 178 samples, 3 cultivars).

- Train/test split, `KNeighborsClassifier` training and evaluation
- **Accuracy: 74.1%**
- Confusion matrix and per-class precision/recall/F1 — class 2 (smallest, 10 samples) is the hardest to separate (F1 = 0.43), highlighting the **class-imbalance problem** with distance-based classifiers

## Key Takeaways

- KNN performance is highly sensitive to feature scaling and `k`
- Minority classes suffer disproportionately — a recurring theme that motivates ensemble methods later in this repo

## Tech Stack

Python · scikit-learn · NumPy · Matplotlib
