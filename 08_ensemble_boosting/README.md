# 08 — Ensemble & Boosting Algorithms

Sequential (boosting) and combination-based (voting/stacking) ensembles — the most powerful classical ML techniques in this repo, building directly on the bagging concepts from `07_random_forest`.

---

## `gradient_boosting.ipynb`

- **Regression**: `GradientBoostingRegressor` → **R² = 0.92**
- **Classification**: `GradientBoostingClassifier` → **Accuracy = 91.3%**

Sequential error-correction (each tree fits the residuals of the previous one) substantially outperforms both the single tree (R² 0.41) and Random Forest results from earlier sections.

## `adaboost.ipynb`

- `AdaBoostClassifier` with `DecisionTreeClassifier` base estimators on synthetically generated data
- **Accuracy = 77.3%**, balanced precision/recall (~0.77 across both classes)

## `xgboost.ipynb`

- `XGBClassifier` on the same dataset as AdaBoost, for direct comparison
- **Accuracy = 88.7%** — a ~11-point improvement over AdaBoost on identical data, demonstrating XGBoost's regularization and second-order optimization advantages

## `voting_and_stacking.ipynb`

Combining heterogeneous models (Decision Tree, Logistic/Linear Regression) via `VotingClassifier`/`VotingRegressor` and `StackingClassifier`/`StackingRegressor`.

| Ensemble | Result |
|---|---|
| Voting Classifier | Accuracy **83.3%** |
| Voting Regressor | R² **0.81** |
| Stacking Classifier | Accuracy **88.0%** |
| Stacking Regressor | R² ≈ **1.00** (train/test) |

Stacking — where a meta-model learns how to combine base-model predictions — outperformed simple voting on both tasks.

## Model Comparison Summary (same/similar data across this section)

| Algorithm | Accuracy |
|---|---|
| AdaBoost | 77.3% |
| Voting Classifier | 83.3% |
| Stacking Classifier | 88.0% |
| XGBoost | 88.7% |

## Key Takeaways

- Boosting (sequential) consistently beat bagging-style ensembles on these datasets
- XGBoost was the strongest single algorithm tested in this repo
- Stacking narrowed the gap to XGBoost using only simple base learners — model *combination strategy* matters as much as model choice

## Tech Stack

Python · scikit-learn · XGBoost · Pandas
