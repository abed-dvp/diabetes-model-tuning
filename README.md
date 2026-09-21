# Diabetes Disease Progression ? Regularization, Hyperparameter Optimization & Model Selection

A machine learning case study in model complexity control, penalized regression, statistical inference vs. predictive modeling, and systematic hyperparameter tuning with a strictly isolated final test partition.

---

## Executive Summary & Methodology

Systematic model selection requires strict separation between exploration, hyperparameter tuning, and final evaluation. When hyperparameter selection touches the test set, reported performance generalizes poorly to production.

This case study establishes a rigorous model selection protocol:
1. **Strict Partitioning**: 80% Training partition (353 samples) for all cross-validation and hyperparameter tuning; 20% Test partition (89 samples) permanently held untouched until final evaluation.
2. **Bias?Variance Diagnosis**: Empirical learning curves and polynomial expansions diagnosing over-fitting and under-fitting regimes.
3. **Penalized Regularization**: Exploring Ridge ($L_2$), Lasso ($L_1$), and ElasticNet ($L_1/L_2$ mixtures) shrinkage paths and automatic feature sparsity.
4. **Statistical Inference vs. Predictive Modeling**: Contrasting classical OLS hypothesis tests ($p$-values) with regularized coefficient shrinkage under severe multicollinearity.
5. **Systematic Hyperparameter Search**: Benchmarking manual Cartesian grid search against `GridSearchCV` and continuous probability distribution sampling (`RandomizedSearchCV` with log-uniform priors).
6. **Support Vector Regression (SVR)**: Deconstructing the $\\epsilon$-insensitive error tube, dual formulation support vector sparsity, and Gaussian RBF kernel parameters.
7. **Pre-Test Model Freezing**: Freezing the optimal model candidate based strictly on CV evidence before evaluating on the held-out test partition.

---

## Evaluation Protocol

```
Training Partition (353 samples)
     ↓
5-Fold Cross-Validation (KFold, shuffle=True, random_state=42)
     ↓
Model Architecture & Hyperparameter Tuning (GridSearch / RandomizedSearch)
     ↓
Candidate Summary Ledger & Performance Auditing
     ↓
Freeze Final Model (Selection based exclusively on CV evidence)
     ↓
Single Final Test Evaluation (Untouched Test Partition, 89 samples)
```

---

## Dataset

Scikit-learn Diabetes benchmark dataset:
- **Observations**: 442 patients
- **Predictors**: 10 numerical features (`age`, `sex`, `bmi`, `bp`, `s1`, `s2`, `s3`, `s4`, `s5`, `s6`)
- **Target**: Quantitative measure of disease progression one year after baseline
- **Splits**: Training (353 samples, 80%) / Held-out Test (89 samples, 20%)

---

## Technical Investigations & Key Findings

### 1. Model Complexity & Bias?Variance Trade-Off
- On `bmi` alone, increasing polynomial degree improved training fit but introduced severe validation variance.
- Expanding across all 10 features, polynomial degree $\\ge 2$ caused rapid validation score collapse, illustrating runaway variance without regularization.

### 2. Regularization Shrinkage Paths
- **Ridge ($L_2$)**: Smoothly damped coefficient magnitudes without enforcing sparsity.
- **Lasso ($L_1$)**: Enforced exact zero coefficients (e.g., zeroing `s2` at $\\alpha=0.05$), acting as an embedded feature selector.
- **Feature Scaling**: Demonstrated why standardized scales are mandatory; unscaled features cause penalties to arbitrarily bias against variables with large natural ranges.

### 3. Statistical Inference vs. Predictive Shrinkage
- Serum lipids `s1` and `s2` exhibit severe multicollinearity ($r > 0.89$), causing OLS to produce erratic, inflated opposing coefficients ($-931.5$ and $+518.1$) with wide confidence intervals.
- Regularization damped these opposing swings, stabilizing weight estimates and preserving generalizability.

### 4. Hyperparameter Search Efficiency
- `GridSearchCV` verified deterministic Cartesian combinations across discrete grids.
- `RandomizedSearchCV` paired with continuous `stats.loguniform` priors discovered high-performing parameter spaces faster and with broader coverage than rigid grids.

### 5. Support Vector Regression (SVR) Mechanics
- Feature standardization dramatically improved SVR fit ($R^2$ improved from $0.3546 \\to 0.4069$).
- $\\epsilon$-tube width controlled support vector sparsity: $100\\%$ of points were support vectors at $\\epsilon=0.1$, dropping to $9.9\\%$ at $\\epsilon=100$.
- Gaussian radius $\\gamma$ controlled local curvature and model flexibility.

---

## Final Model Selection & Test Evaluation

Candidate selection was locked prior to opening the test partition. **Lasso Regression ($L_1$)** achieved the highest cross-validation score and was selected as the final production candidate.

| Metric / Attribute | Value |
|:---|:---|
| **Selected Architecture** | **Lasso Regression ($L_1$)** |
| **Optimal Hyperparameters** | `alpha = 0.01, max_iter = 10000` |
| **Feature Representation** | Pre-scaled scikit-learn features (`X_scaled_train`) |
| **Mean 5-Fold CV $R^2$** | **0.4810** |
| **CV $R^2$ Standard Deviation** | **0.0396** |
| **Final Test $R^2$** | **0.4567** |
| **Final Test MAE** | **42.8318** |
| **Final Test RMSE** | **53.6522** |
| **Generalization Difference (Test $R^2$ - CV $R^2$)** | **-0.0243** |

*Methodological note: Model selection was performed exclusively on cross-validation evidence before unsealing the test partition. The test $R^2$ of 0.4567 aligns closely with cross-validation expectations (delta of -0.0243), confirming strong out-of-sample generalization without data leakage.*

---

## Running Locally

```bash
git clone https://github.com/abed-dvp/diabetes-model-tuning.git
cd diabetes-model-tuning

pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

---

## Project Context
This case study documents advanced model tuning practices covering parameter regularization, cross-validation architectures, and support vector machines.
