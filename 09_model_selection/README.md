# 09 — Model Selection & Hyperparameter Tuning

The final step in the ML workflow: systematically choosing the best model configuration rather than guessing.

---

## `gridsearchcv_hyperparameter_tuning.ipynb`

`GridSearchCV` over an `SVC` on the **Iris dataset**, tuning `C` and `kernel`.

| Params | CV Accuracy |
|---|---|
| `{'C': 1, 'kernel': 'linear'}` | 0.980 (±0.033) |
| `{'C': 1, 'kernel': 'rbf'}` | 0.967 (±0.042) |
| `{'C': 10, 'kernel': 'linear'}` | 0.973 (±0.078) |
| `{'C': 10, 'kernel': 'rbf'}` | 0.980 (±0.033) |

## Key Takeaways

- Cross-validated grid search surfaces not just the best mean score but also its **variance** — `{'C': 10, 'kernel': 'linear'}` ties on mean accuracy but has more than double the standard deviation, making it a less reliable choice than `{'C': 1, 'kernel': 'linear'}`
- This closes the loop on the repo: every algorithm in `03`–`08` can be slotted into this same `GridSearchCV` workflow for production-grade tuning

## Tech Stack

Python · scikit-learn
