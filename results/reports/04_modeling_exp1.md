# Stage 4 Modeling Report — Experiment 1

## Description
Global model WITH gender feature (`Gender_encoded`), evaluated on:
- Full test set
- Male subgroup
- Female subgroup

## Logistic Regression
| Split | N | Accuracy | Precision | Recall | F1 | ROC-AUC | Recall 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| Full | 117 | 0.6752 | 0.9592 | 0.5663 | 0.7121 | 0.7835 | — |
| Male | 89 | 0.7079 | 0.9535 | 0.6308 | 0.7593 | 0.8288 | — |
| Female | 28 | 0.5714 | 1.0000 | 0.3333 | 0.5000 | 0.6167 | [0.118, 0.572] |

## Random Forest
| Split | N | Accuracy | Precision | Recall | F1 | ROC-AUC | Recall 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| Full | 117 | 0.7179 | 0.7604 | 0.8795 | 0.8156 | 0.7849 | — |
| Male | 89 | 0.7753 | 0.7848 | 0.9538 | 0.8611 | 0.8112 | — |
| Female | 28 | 0.5357 | 0.6471 | 0.6111 | 0.6286 | 0.7000 | [0.437, 0.882] |

## XGBoost
| Split | N | Accuracy | Precision | Recall | F1 | ROC-AUC | Recall 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| Full | 117 | 0.7350 | 0.8023 | 0.8313 | 0.8166 | 0.7860 | — |
| Male | 89 | 0.7528 | 0.8116 | 0.8615 | 0.8358 | 0.7872 | — |
| Female | 28 | 0.6786 | 0.7647 | 0.7222 | 0.7429 | 0.7389 | [0.500, 0.933] |

## Key Notes
- Best overall model on full test set: **XGBoost** (F1=0.8166, AUC=0.7860).
- Male recall is higher than female recall for all models (male-favored gap).
- Small female test sample size (`n=28`) is reflected in wider recall CI.
