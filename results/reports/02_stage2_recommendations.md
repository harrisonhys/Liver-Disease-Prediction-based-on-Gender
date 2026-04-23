# Stage 2 → Stage 3 Recommendations

Generated from deep analysis of EDA outputs before proceeding to data preparation.

---

## Key Findings that Drive These Recommendations

### 1. Double-Imbalance (Critical)
Female No-Liver-Disease patients = **50** (8.6% of total dataset).
A naive 80/20 split leaves ~10 healthy female patients in the test set — too few for stable subgroup metrics.

### 2. Feature Importance Differs by Gender
Correlation rankings against the binary target:

| Rank | Male | Female |
|---|---|---|
| 1 | DB (0.247) | **Alkphos (0.244)** |
| 2 | TB (0.217) | DB (0.223) |
| 3 | ALB (−0.187) | TB (0.215) |
| 4 | A/G Ratio (−0.168) | Sgot (0.215) |
| 5 | Alkphos (0.163) | Sgpt (0.193) |

Alkphos is rank #1 for females but only #5 for males — direct evidence supporting Hypothesis H2.

### 3. Weak Biomarker Separation in Females
Median DB for female Liver Disease = 0.20, same as female No Liver Disease.
For males the same ratio is 3.75×. Models trained on pooled data will carry the male separation pattern and risk missing female cases.

### 4. Asymmetric Extreme Outliers
99th percentile Sgot: Male = 1014, Female = 614. Max Sgot male = 4929 (8× female max).
StandardScaler will be distorted by male outliers and compress female feature variance.

---

## Mandatory Preprocessing Decisions for Stage 3

### M1 — Stratified Split by Gender × Target (4-class)
Use a combined `Gender_Target` stratification key instead of splitting on target alone.
This guarantees all four cells (Male+/Male−/Female+/Female−) are proportionally represented in both train and test sets.

### M2 — RobustScaler Instead of StandardScaler
Enzyme and bilirubin features are heavily right-skewed with extreme outliers.
RobustScaler (median + IQR) is resistant to these extremes and avoids suppressing female feature variance.

### M3 — Log Transformation Before Scaling
Apply `log1p` to TB, DB, Sgpt, Sgot, Alkphos before scaling.
These features are right-skewed. Log compression reduces the influence of extreme outliers and improves linear separability.

### M4 — Preserve Gender Column Separately
After label-encoding Gender for model input, keep the original string column in a separate tracking DataFrame.
This column is needed at evaluation time for every subgroup split and must not be lost in the pipeline.

---

## Recommended Value-Adds for the Paper

### V1 — Engineered Feature: DB/TB Ratio (Fractional Bilirubin)
`fractional_bilirubin = DB / (TB + 1e-6)`
This is a clinically established liver function indicator. It may be more stable and discriminative for females than raw DB or TB alone. Include it as an additional feature and monitor its importance in SHAP.

### V2 — Per-Subgroup Descriptive Statistics Table
Compute `describe()` separately for Male Positive, Male Negative, Female Positive, Female Negative.
Save as `results/tables/03_subgroup_descriptive_stats.csv`. This table can be inserted directly into the paper's Data Analysis section.

### V3 — Bootstrap Confidence Intervals for Subgroup Metrics (Stage 4)
With n=50 for the Female Negative group, a single point estimate (e.g. recall = 0.62) is not sufficient for academic claims.
In Stage 4, compute 95% CI via 1000-iteration bootstrap for each subgroup metric. Report as `metric ± CI`.

---

## What NOT to Do

- **Do not apply SMOTE** at this stage. The 2.5:1 class imbalance is manageable without oversampling.
  If used later, it must be applied inside the training fold only (never on the full dataset before splitting) and justified explicitly in the paper.
- **Do not drop outliers**. Extreme enzyme values are clinically meaningful for liver disease detection.
  Use robust preprocessing instead of removal.
- **Do not use one-hot encoding for Gender**. Binary label encoding (Male=1, Female=0) is sufficient
  and avoids dummy-variable redundancy.
