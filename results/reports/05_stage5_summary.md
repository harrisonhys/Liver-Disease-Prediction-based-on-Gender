# Stage 5 Summary — Evaluation and Fairness Analysis

## Scope
Stage 5 evaluates model performance and fairness using outputs from Stage 4 across three experiment settings:
- `exp1`: global model with gender feature
- `exp2`: global model without gender feature
- `exp3`: separate subgroup models

Metrics emphasized in this stage:
- Accuracy gap (male - female)
- F1 gap (male - female)
- Recall gap (male - female)
- False Negative Rate (FNR) per gender and FNR gap (female - male)
- False positive vs false negative profile from confusion matrices

## Main Findings

### 1. Best Global Performance
Best global setup by full-test F1 is:
- `exp2 / XGBoost`
- Full Accuracy = 0.7692
- Full F1 = 0.8402
- Full AUC = 0.7782

### 2. Fairness Disparity is Consistent
All evaluated configurations show male-favored recall (`male_recall > female_recall`).

In `exp1`:
- Logistic Regression recall gap = +0.2975
- Random Forest recall gap = +0.3427
- XGBoost recall gap = +0.1393 (smallest in exp1)

### 3. Female Under-Diagnosis Risk (FNR)
Female FNR is higher than male FNR in every compared setup.

Most critical case:
- `exp3 / LogisticRegression`
- Female FNR = 0.7222

Best female FNR observed:
- `exp3 / RandomForest`
- Female FNR = 0.1111

### 4. Effect of Removing Gender Feature
Comparing `exp2` against `exp1`, disparity remains present. This indicates fairness issues are not driven only by explicit gender feature usage, but also by subgroup-specific clinical feature distributions.

### 5. Error-Type Pattern (Clinical View)
- Some models reduce FN but increase FP substantially.
- In clinical screening, reducing FN is usually more important than reducing FP, because FN means missed disease detection and delayed treatment.

## Key Numerical Snapshot

| Experiment | Model | Recall Gap (M-F) | Female FNR | Male FNR |
|---|---|---:|---:|---:|
| exp1 | LogisticRegression | +0.2975 | 0.6667 | 0.3692 |
| exp1 | RandomForest | +0.3427 | 0.3889 | 0.0462 |
| exp1 | XGBoost | +0.1393 | 0.2778 | 0.1385 |
| exp2 | LogisticRegression | +0.2975 | 0.6667 | 0.3692 |
| exp2 | RandomForest | +0.3427 | 0.3889 | 0.0462 |
| exp2 | XGBoost | +0.2410 | 0.3333 | 0.0923 |
| exp3 | LogisticRegression | +0.3684 | 0.7222 | 0.3538 |
| exp3 | RandomForest | +0.0649 | 0.1111 | 0.0462 |
| exp3 | XGBoost | +0.1299 | 0.2222 | 0.0923 |

## Stage-5 Decision Notes
- Primary global reference model: `exp2 / XGBoost` (best full-test score profile).
- Primary fairness-sensitive candidate: `exp3 / RandomForest` (lowest female FNR, smallest recall gap in subgroup setting).
- Stage 6 should explicitly communicate the trade-off between overall global performance and subgroup fairness risk.

## Produced Stage-5 Artifacts
- `results/reports/05_evaluation_comparative_analysis.txt`
- `results/reports/05_ai_ready_evaluation_summary.txt`
- `results/tables/05_model_comparison.csv`
- `results/tables/05_confusion_by_group.csv`
- `results/reports/05_stage5_summary.md`
