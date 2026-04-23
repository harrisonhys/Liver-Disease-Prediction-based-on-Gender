# Stage 4 — Consolidated Modeling Summary

## Scope
Stage 4 covers three scenarios:
1. **Experiment 1**: Global model with gender feature
2. **Experiment 2**: Global model without gender feature
3. **Experiment 3**: Separate models by gender subgroup

Models evaluated in every scenario:
- Logistic Regression
- Random Forest
- XGBoost

Evaluation views:
- Full test set
- Male subgroup
- Female subgroup

---

## Best Overall Model (Experiment 1, Full Test Set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6752 | 0.9592 | 0.5663 | 0.7121 | 0.7835 |
| Random Forest | 0.7179 | 0.7604 | 0.8795 | 0.8156 | 0.7849 |
| **XGBoost** | **0.7350** | 0.8023 | 0.8313 | **0.8166** | **0.7860** |

Best by balanced performance (F1 + AUC): **XGBoost**.

---

## Fairness Signal — Recall Gap (Male − Female) in Experiment 1
| Model | Male Recall | Female Recall | Gap |
|---|---:|---:|---:|
| Logistic Regression | 0.6308 | 0.3333 | +0.2975 |
| Random Forest | 0.9538 | 0.6111 | +0.3427 |
| XGBoost | 0.8615 | 0.7222 | +0.1393 |

Interpretation:
- All gaps are positive, indicating **male-favored recall**.
- XGBoost has the smallest recall gap among the three.

---

## Effect of Removing Gender Feature (Experiment 2 vs Experiment 1)
Female recall change:

| Model | Exp1 Female Recall | Exp2 Female Recall | Delta |
|---|---:|---:|---:|
| Logistic Regression | 0.3333 | 0.3333 | +0.0000 |
| Random Forest | 0.6111 | 0.6111 | +0.0000 |
| XGBoost | 0.7222 | 0.6667 | -0.0555 |

Interpretation:
- For LR and RF, removing explicit gender does not reduce the recall gap.
- For XGBoost, female recall declines slightly.
- Bias pattern likely also comes from subgroup distribution differences in clinical features.

---

## Effect of Separate Subgroup Models (Experiment 3 vs Experiment 1)
Female recall improvement:

| Model | Exp1 Female Recall | Exp3 Female Recall | Delta |
|---|---:|---:|---:|
| Logistic Regression | 0.3333 | 0.2778 | -0.0555 |
| Random Forest | 0.6111 | 0.8889 | +0.2778 |
| XGBoost | 0.7222 | 0.7778 | +0.0556 |

Interpretation:
- **Random Forest** benefits most from female-specific training.
- XGBoost also improves female recall in separate modeling.
- Logistic Regression is unstable for the small female subgroup.

---

## Bootstrap CI Note
Female recall is reported with 95% bootstrap CI (`n=28` female test samples), so uncertainty is explicitly reflected for subgroup conclusions.

---

## Files Produced in Stage 4
- `results/tables/04_model_metrics.csv`
- `results/reports/04_modeling_exp1.txt`
- `results/reports/04_modeling_exp2.txt`
- `results/reports/04_modeling_exp3.txt`
- `results/reports/04_ai_ready_modeling_summary.txt`
- `results/reports/04_modeling_exp1.md`
- `results/reports/04_modeling_exp2.md`
- `results/reports/04_modeling_exp3.md`
- `results/reports/04_stage4_summary.md`
- 39 plots in `results/plots/04_*.png`
- 12 trained model files in `results/models/*.pkl`

---

## Recommendation Before Stage 5
- Use **XGBoost (Exp1)** as global reference model.
- Use **Random Forest (Exp3 female)** as subgroup-sensitive benchmark.
- In Stage 5, prioritize fairness metrics:
  - Accuracy gap
  - F1 gap
  - Recall gap
  - Female false negative rate
- Report both point estimate and CI for female subgroup metrics.
