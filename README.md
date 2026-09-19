# Diabetes Model Tuning

## Project Goal

Diagnose model complexity and systematically tune regression models using validation-based model selection while preserving a truly untouched final Test Set.

## Dataset

Scikit-learn Diabetes dataset:
- **442 observations**
- **10 numerical input features** (age, sex, bmi, bp, s1, s2, s3, s4, s5, s6)
- **Continuous target:** Quantitative measure of diabetes disease progression one year after baseline

### Partitioning:
- **Training Set:** 353 observations (80%)
- **Test Set:** 89 observations (20%) — permanently segregated and held strictly locked until final model evaluation.

## Evaluation Protocol

```
Training Partition (353 samples)
     ↓
5-Fold Cross-Validation (KFold, shuffle=True, random_state=42)
     ↓
Model Architecture & Hyperparameter Tuning
     ↓
Candidate Summary Ledger
     ↓
Freeze Final Model (Selection based exclusively on CV evidence)
     ↓
Single Final Test Evaluation (Untouched Test Partition, 89 samples)
```

The permanent Test Set was not used during any tuning, search, kernel selection, or hyperparameter decision.

## Project Learning Flow

1. **Baseline & Evaluation Protocol:** Load raw and pre-scaled representations; establish permanent 80/20 split; establish 5-fold cross-validation; evaluate OLS baseline ($R^2 \approx 0.4804$).
2. **Model Complexity & Learning Curves:** Polynomial feature expansion (degrees 1–10 on BMI, degrees 1–3 across all features); bias–variance tradeoff diagnosis; learning curve analysis on simple vs. complex models.
3. **Ridge & Lasso Regularization:** Regularized loss formulation; Ridge ($L_2$) continuous shrinkage paths; Lasso ($L_1$) feature selection and exact zero coefficients; feature scaling demonstration.
4. **ElasticNet & Statistical Inference:** ElasticNet $L_1/L_2$ mixing ratio (`l1_ratio`); `statsmodels` OLS with `sm.add_constant`; $p$-values and statistical significance vs. regularization coefficient shrinkage under collinearity.
5. **Hyperparameter Search & SVR:** Manual Cartesian grid search (`itertools.product`); systematic search with `GridSearchCV` (`n_jobs=-1`); continuous distribution sampling (`stats.uniform`, `stats.norm`, `stats.loguniform`, `.rvs()`) with `RandomizedSearchCV`; Coarse $\to$ Fine search resolution; Support Vector Regression (SVR) margin tube, scaling sensitivity, linear support vectors extraction, Linear/Polynomial/RBF kernels, and $C$, $\epsilon$, $\gamma$ (myopia factor) parameter experiments.
6. **Final Model Selection & Test Evaluation:** Rebuild CV candidate ledger; freeze selection rule; freeze final model; refit on full training partition; single test set evaluation; final model parameters and complete lesson/scope audits.

## What This Project Demonstrates

- **Parameters vs. Hyperparameters:** Distinction between weights learned by optimization and parameters governing complexity.
- **Model Complexity:** Controlling underfitting (excess bias) and overfitting (excess variance).
- **Bias–Variance Tradeoff & Irreducible Error:** Diagnosing error components using training and cross-validation curves.
- **Learning Curves:** Analyzing performance as training sample size scales.
- **Regularization ($L_1, L_2$, ElasticNet):** Penalized loss formulations, shrinkage paths, and sparsity.
- **Feature Scaling Before Regularization & SVR:** Why uniform scale is required for penalty and distance metrics.
- **Statistical Inference vs. Predictive Modeling:** Classical hypothesis testing ($p$-values) vs. regularization under multicollinearity.
- **Hyperparameter Search Techniques:** Exhaustive grid search, randomized continuous sampling, and coarse-to-fine zooming.
- **Support Vector Regression (SVR):** Hyperplanes, $\epsilon$-insensitive loss tubes, support vectors, convex optimization, and kernel trick.
- **SVR Hyperparameters:** Regularization ($C$), margin sparsity ($\epsilon$), and Gaussian radius ($\gamma$ / myopia factor).
- **Methodological Integrity:** Rigid separation between training/validation tuning and solitary final test evaluation.

## Key Findings

- **Complexity Control:** Increasing polynomial degree on BMI improved training fit ($R^2 > 0.60$) but severely degraded cross-validation generalization ($R^2 < 0$), confirming severe overfitting.
- **Learning Curves:** Visualized how simple models plateau early, while complex models exhibit a wide generalization gap that narrows as training size grows.
- **Regularization Behavior:** Ridge ($L_2$) smoothly shrunk all coefficients without sparsity; Lasso ($L_1$) drove coefficients to exact zeros (e.g., `s2` zeroed at $\alpha=0.05$), performing embedded feature selection.
- **Statistical Inference:** Serum lipids `s1` and `s2` exhibited high collinearity ($r > 0.89$), causing OLS to assign large opposing coefficients ($-931.5$ and $+518.1$). Regularization damped these opposing swings to protect generalization.
- **Hyperparameter Search:** `GridSearchCV` perfectly replicated manual `itertools` search. `RandomizedSearchCV` sampled continuous `stats.loguniform` priors efficiently, identifying high-performing parameter combinations.
- **SVR Properties:** Scaling materially improved SVR performance ($0.3546 \to 0.4069$). $\epsilon$ governed support vector sparsity ($100\%$ at $\epsilon=0.1$ down to $9.9\%$ at $\epsilon=100$), and $\gamma$ controlled local Gaussian reach (myopia factor).
- **Model Selection & Test Generalization:** Leading linear models yielded closely clustered CV scores ($0.4804 - 0.4810$). Model selection was executed prior to opening the test set; the frozen model achieved an independent Test $R^2$ of $0.4567$, closely matching cross-validation expectations.

## Final Model Selection

| Metric / Attribute | Value |
|:---|:---|
| **Selected Model** | **Lasso Regression ($L_1$)** |
| **Frozen Hyperparameters** | `alpha = 0.01, max_iter = 10000` |
| **Feature Representation** | scikit-learn pre-scaled features (`X_scaled_train`) |
| **Mean 5-Fold CV $R^2$** | **0.4810** |
| **CV $R^2$ Std** | **0.0396** |
| **Final Test $R^2$** | **0.4567** |
| **Final Test MAE** | **42.8318** |
| **Final Test RMSE** | **53.6522** |
| **Generalization Difference** | **-0.0243** (well within $1\sigma$ CV std of $0.0396$) |

*Note: Final model selection was made strictly based on cross-validation evidence before the Test Set was opened.*

## Lesson Coverage

**63 / 63 (100.0%)** — Verified programmatically:
- Foundations: 4 / 4
- Model Complexity: 10 / 10
- Overfitting Strategies: 6 / 6
- Regularization: 10 / 10
- Statistical Inference: 5 / 5
- Hyperparameter Search: 14 / 14
- SVM / SVR: 14 / 14

## Scope Boundaries

This is a strictly lesson-locked educational machine learning project aligned with the Model Tuning curriculum.
The project intentionally excludes:
- Ensemble methods (Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost)
- Neural networks (PyTorch, TensorFlow, Keras, MLP)
- Deployment, web frameworks, or APIs (Flask, FastAPI, Streamlit, Gradio)
- Advanced external hyperparameter optimizers (Optuna, Hyperopt, Bayesian Optimization)
- Model interpretability packages (SHAP)
- Nested cross-validation or VIF collinearity diagnostics
- Classification algorithms (SVC) or artificial classification targets

## How to Run

Clone the repository and install the minimal dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook top-to-bottom:

```bash
jupyter notebook notebook.ipynb
```

## Status

**Completed — Model Tuning lesson project**
