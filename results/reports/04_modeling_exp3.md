# Stage 4 Modeling Report — Experiment 3

## Description
Separate models per gender subgroup (no `Gender_encoded` in input features):
- Male-only model trained on male train split, evaluated on male test split
- Female-only model trained on female train split, evaluated on female test split

## Logistic Regression
| Split | N | Accuracy | Precision | Recall | F1 | ROC-AUC | Recall 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| Male | 89 | 0.7079 | 0.9333 | 0.6462 | 0.7636 | 0.8179 | — |
| Female | 28 | 0.5000 | 0.8333 | 0.2778 | 0.4167 | 0.5611 | [0.077, 0.500] |

## Random Forest
| Split | N | Accuracy | Precision | Recall | F1 | ROC-AUC | Recall 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| Male | 89 | 0.7753 | 0.7848 | 0.9538 | 0.8611 | 0.7917 | — |
| Female | 28 | 0.7500 | 0.7619 | 0.8889 | 0.8205 | 0.7556 | [0.824, 1.000] |

## XGBoost
| Split | N | Accuracy | Precision | Recall | F1 | ROC-AUC | Recall 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| Male | 89 | 0.7303 | 0.7662 | 0.9077 | 0.8310 | 0.7981 | — |
| Female | 28 | 0.6786 | 0.7368 | 0.7778 | 0.7568 | 0.7778 | [0.571, 0.944] |

## Key Notes
- Female-specific **Random Forest** shows strongest gain vs global model (female recall 0.8889).
- Female-specific XGBoost also improves female recall vs Exp1 (0.7778 vs 0.7222).
- Logistic Regression degrades in female-specific setup, indicating linear model underfitting on small female subset.
