# 07 — Random Forest & Bagging

Ensembling decision trees to reduce variance — the direct follow-up to the overfitting issues seen in `05_decision_trees`.

---

## `random_forest_bagging_titanic.ipynb`

Classification on the **Titanic dataset**, comparing a single tree against bagged ensembles.

| Model | Result |
|---|---|
| Single `DecisionTreeClassifier` | Train accuracy **87.2%**, Test accuracy **78.4%** — clear overfitting gap |
| `RandomForestClassifier` (OOB) | **OOB score 81.7%**, Test accuracy **81.7%** — OOB estimate matches held-out test almost exactly |
| `BaggingClassifier` + Decision Tree | Accuracy **77.2%** |
| `BaggingClassifier` + Logistic Regression | Accuracy **79.5%** |

## Key Takeaways

- Random Forest closed the train/test gap seen with a single tree (87.2% → 81.7% test, vs 78.4% for the single tree) — **bagging reduces variance without hurting bias much**
- The **out-of-bag (OOB) score** is a free, built-in validation estimate that closely tracked the true test accuracy — useful when data is limited
- Bagging the *same* base estimator (Decision Tree) underperformed the Random Forest, showing the value of Random Forest's added **feature randomness** at each split, not just bootstrapping rows

## Tech Stack

Python · scikit-learn · Pandas
