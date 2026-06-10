# Machine Learning

Classical machine learning, organized as a structured learning path — from data exploration through statistics, individual algorithms, ensembles, and hyperparameter tuning. Built during a PGDM in AI & Data Science, applied to real datasets (Titanic, Iris, Wine, Diabetes, California Housing, Digits) with measured results, not just textbook code.

---

## Learning Path

| # | Topic | Core Question Answered |
|---|---|---|
| [01](01_eda_and_feature_engineering) | **EDA & Feature Engineering** | Is the data clean, and what features actually matter? |
| [02](02_statistics) | **Statistical Foundations** | Does the data meet the assumptions my models rely on? |
| [03](03_knn) | **K-Nearest Neighbors** | What's the simplest possible baseline? |
| [04](04_linear_regression) | **Linear Regression** | How much variance can a linear model explain? |
| [05](05_decision_trees) | **Decision Trees** | Can a single rule-based model do better — and how do I stop it from overfitting? |
| [06](06_svm) | **Support Vector Machines** | How do margin-based methods perform on tabular vs image data? |
| [07](07_random_forest) | **Random Forest & Bagging** | Does ensembling reduce the overfitting seen in a single tree? |
| [08](08_ensemble_boosting) | **Ensemble & Boosting** | What's the strongest classical model achievable, and how do they compare? |
| [09](09_model_selection) | **Model Selection** | How do I systematically pick the best configuration? |

Each folder has its own README with techniques used, datasets, and measured results.

---

## Headline Results

| Dataset / Task | Best Result | Model |
|---|---|---|
| Titanic (survival classification) | **81.7%** accuracy (OOB-validated) | Random Forest |
| Iris (species classification) | **93.3%** accuracy | SVM (tuned `C`) |
| Diabetes (progression regression) | **R² = 0.92** | Gradient Boosting |
| Synthetic binary classification | **88.7%** accuracy | XGBoost |
| Wine (cultivar classification) | **74.1%** accuracy | KNN |
| Digits (image classification, 10 classes) | Near-perfect confusion matrix | SVM |

---

## Key Concepts Demonstrated

Supervised Learning · Classification & Regression · Bias-Variance Tradeoff · Overfitting & Pruning · Bagging vs Boosting · Voting & Stacking Ensembles · Cross-Validation · Hyperparameter Tuning (`GridSearchCV`) · Feature Engineering · Statistical Distributions

## Tech Stack

Python · scikit-learn · XGBoost · Pandas · NumPy · Matplotlib · SciPy · Jupyter Notebook
