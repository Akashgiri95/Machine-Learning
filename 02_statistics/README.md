# 02 — Statistical Foundations

The probability and distribution concepts that underpin model assumptions, hypothesis testing, and data normalization.

---

## `statistical_distributions.ipynb`

- Generating and visualizing the **Normal (Gaussian) distribution** from first principles
- **Kernel Density Estimation (KDE)** vs histogram-based density estimates
- **Q-Q plots** to visually test whether sample data follows a normal distribution
- Validated empirical statistics against theoretical parameters — e.g. empirical mean 5.039 / std 1.958 against an expected μ=5, σ=2

## Why this matters

Many ML algorithms (linear regression, LDA, Gaussian Naive Bayes) assume normally-distributed features or residuals. Being able to test that assumption — and recognize when it's violated — directly informs preprocessing decisions like transformation or scaling used in later sections.

## Tech Stack

Python · NumPy · SciPy · Matplotlib
