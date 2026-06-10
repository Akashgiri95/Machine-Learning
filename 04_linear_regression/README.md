# 04 — Linear Regression

The foundational regression algorithm — establishing baselines that tree-based and ensemble regressors are later compared against.

---

## `linear_regression_fundamentals.ipynb`

Worked examples of `sklearn.linear_model.LinearRegression` on simple synthetic and toy datasets.

- Single-feature regression with coefficient interpretation (e.g. coefficient ≈ 938.2)
- **MSE: 2548.07, R²: 0.47** on a toy diabetes-style dataset
- Visualizing the fitted line against actual data points

## `linear_regression_california_housing.ipynb`

Regression on the **California Housing dataset** (20,640 samples, 8 features) — the modern scikit-learn replacement for the deprecated `load_boston`.

- Dataset loading via `fetch_california_housing` and feature/target inspection
- Train/test split (13,828 / 6,812)
- Notes on why `load_boston` was removed from scikit-learn (ethical concerns with the dataset) and the migration path

## Key Takeaways

- A single-feature linear model explains less than half the variance (R² = 0.47) — motivating the move to richer feature sets and non-linear models in later sections
- Practical experience handling scikit-learn dataset deprecations

## Tech Stack

Python · scikit-learn · NumPy · Matplotlib
