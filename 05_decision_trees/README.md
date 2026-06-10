# 05 — Decision Trees

Tree-based models for both classification and regression, with an emphasis on **pruning** as a tool against overfitting.

---

## `decision_tree_classifier_iris.ipynb`

Two worked examples of `DecisionTreeClassifier` on the Iris dataset, with confusion matrices and classification reports — establishing the baseline tree-fitting workflow used throughout this section.

## `decision_tree_classifier_titanic_pruning.ipynb`

Classification on the **Titanic dataset**, focused on **hyperparameter tuning to control overfitting**:

- **Pre-pruning**: swept `max_depth` (5–10) and `min_samples_split` (5–30), tracking accuracy at each setting
- **Post-pruning**: cost-complexity pruning via `ccp_alpha`
- Best configuration reached **82.5% accuracy** (`min_samples_split=5` or `10`) — a meaningful jump over the unconstrained tree (~77–80%)

## `decision_tree_regressor_diabetes.ipynb`

Regression on the **Diabetes dataset** (`load_diabetes`) using `DecisionTreeRegressor(max_depth=7, min_samples_leaf=20)`.

- **Train R² = 0.55, Test R² = 0.41** (Train MSE = 2795.7, Test MSE = 3177.7)
- The train/test gap illustrates a tree that is still mildly overfitting even with depth and leaf-size constraints — the motivation for **Random Forest** (next section)

## Key Takeaways

- Pruning hyperparameters (`max_depth`, `min_samples_split`, `ccp_alpha`) directly trade off bias vs variance
- A single tree's instability on the Diabetes dataset sets up the case for ensembling

## Tech Stack

Python · scikit-learn · Pandas · Matplotlib
