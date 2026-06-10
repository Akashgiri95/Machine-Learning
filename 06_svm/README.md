# 06 — Support Vector Machines

Margin-based classifiers and regressors, covering linear and RBF kernels, hyperparameter tuning, and a high-dimensional image classification task.

---

## `svm_fundamentals.ipynb`

Introductory examples of `sklearn.svm.SVC` — kernel selection (`rbf`, `linear`), the role of `C`, `gamma`, and `degree` — building intuition before applying SVMs to real datasets.

## `svm_classification_iris.ipynb`

Classification on the **Iris dataset** with `SVC`.

- Baseline accuracy: **91.1%**
- Swept `C` ∈ {0.5, 1, 2, 3, 4, 5} — best accuracy **93.3%** at `C=1, 4, 5`, showing the model is fairly robust to `C` in this range

## `svm_digit_classification.ipynb`

Multi-class classification on **`load_digits`** (8×8 grayscale digit images, 10 classes).

- Confusion matrix shows near-perfect separation across all 10 digit classes (≥85/90 correct per class)
- Demonstrates SVM's strength on high-dimensional, well-separated image data

## `svm_regression_diabetes.ipynb`

Regression on the **Diabetes dataset** comparing `SVR` and `LinearSVR`.

- Baseline `SVR`: **train R² = 0.66, test R² = 0.49**
- `GridSearchCV` over kernel/`C`/`epsilon` → best params `{'C': 10, 'epsilon': 0.1, 'kernel': 'linear'}`
- Simplified `LinearSVR`: **test R² = 0.47** — close to the tuned full SVR at a fraction of the compute cost

## `SVM-Classification.pdf`

Reference notes on the SVC parameter space and decision function shapes.

## Key Takeaways

- SVMs are highly competitive on image data (`load_digits`) but only moderately strong on tabular regression (Diabetes) compared to ensembles
- `GridSearchCV` tuning gave a marginal gain over a well-chosen `LinearSVR` — not always worth the extra compute

## Tech Stack

Python · scikit-learn · NumPy · Matplotlib
