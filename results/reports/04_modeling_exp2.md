# Stage 4 Modeling Report — Experiment 2

## Description
Global model WITHOUT gender feature (`Gender_encoded` removed), evaluated on:
- Full test set
- Male subgroup
- Female subgroup

## Logistic Regression
| Split | N | Accuracy | Precision | Recall | F1 | ROC-AUC | Recall 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| Full | 117 | 0.6752 | 0.9592 | 0.5663 | 0.7121 | 0.7838 | — |
| Male | 89 | 0.7079 | 0.9535 | 0.6308 | 0.7593 | 0.8288 | — |
| Female | 28 | 0.5714 | 1.0000 | 0.3333 | 0.5000 | 0.6167 | [0.117, 0.571] |

## Random Forest
| Split | N | Accuracy | Precision | Recall | F1 | ROC-AUC | Recall 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| Full | 117 | 0.7179 | 0.7604 | 0.8795 | 0.8156 | 0.7826 | — |
| Male | 89 | 0.7753 | 0.7848 | 0.9538 | 0.8611 | 0.8119 | — |
| Female | 28 | 0.5357 | 0.6471 | 0.6111 | 0.6286 | 0.6417 | [0.357, 0.842] |

## XGBoost
| Split | N | Accuracy | Precision | Recall | F1 | ROC-AUC | Recall 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| Full | 117 | 0.7692 | 0.8256 | 0.8554 | 0.8402 | 0.7782 | — |
| Male | 89 | 0.7978 | 0.8310 | 0.9077 | 0.8676 | 0.7705 | — |
| Female | 28 | 0.6786 | 0.8000 | 0.6667 | 0.7273 | 0.7278 | [0.444, 0.889] |

## Key Notes
- Removing gender feature has limited effect on female recall for Logistic Regression and Random Forest.
- For XGBoost, female recall decreases from 0.7222 (Exp1) to 0.6667 (Exp2).
- This suggests disparity is not explained only by explicit gender feature usage; distribution shift across lab features remains influential.
