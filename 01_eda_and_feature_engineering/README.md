# 01 — Exploratory Data Analysis & Feature Engineering

Every ML pipeline starts here: understanding the data before modelling it. This section covers structured EDA on the Iris dataset and a full data-cleaning + feature-engineering pass on the Titanic dataset.

---

## Notebooks

### `eda_iris_questions.ipynb` / `eda_iris_solutions.ipynb`

Structured EDA exercise on **`Iris.csv`** / **`Iris_WithNullOutlier.csv`** (a deliberately corrupted version with nulls and outliers).

- Inspecting shape, dtypes, and column descriptions — and reasoning about *why* `.describe()` excludes non-numeric columns
- Re-indexing, row selection, and class-wise sample counts
- Detecting and handling **null values and outliers** before they corrupt downstream statistics

### `feature_engineering_titanic.ipynb`

End-to-end data cleaning and feature engineering on the **Titanic dataset** (891 rows × 12 columns).

- **Missing value analysis** — `Age` (177 missing), `Cabin`, `Embarked` (2 missing) — and imputation strategy per column
- **Duplicate and inconsistency checks** across categorical fields
- **Feature engineering**: derived `Deck` from `Cabin`, cleaned string columns, encoded categoricals
- Final **train/test split**: 712 / 179 rows, expanded to 22 features after encoding

---

## Datasets

| Dataset | Source | Used for |
|---|---|---|
| `Iris.csv`, `Iris_WithNullOutlier.csv` | UCI Iris (clean + corrupted variant) | EDA fundamentals, data quality checks |
| Titanic | Kaggle Titanic | Missing-value handling, feature engineering |

## Tech Stack

Python · Pandas · NumPy · Matplotlib
