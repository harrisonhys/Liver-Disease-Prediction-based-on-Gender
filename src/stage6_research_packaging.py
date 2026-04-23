from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import PLOTS_DIR, REPORTS_DIR, TABLES_DIR, ensure_directories


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _load_stage5_table() -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / "05_model_comparison.csv")


def _best_global(df: pd.DataFrame) -> pd.Series:
    global_df = df[df["experiment"].isin(["exp1", "exp2"])].copy()
    return global_df.sort_values("full_f1", ascending=False).iloc[0]


def _best_fairness(df: pd.DataFrame) -> pd.Series:
    return df.sort_values(["female_fnr", "recall_gap_male_minus_female"]).iloc[0]


def _stage6_executive_summary(df: pd.DataFrame) -> str:
    best_global = _best_global(df)
    best_fair = _best_fairness(df)

    lines = [
        "Stage 6 - Executive Summary",
        "",
        "Project title:",
        "Analysis of Machine Learning Model Performance Differences in Liver Disease Prediction Based on Gender",
        "",
        "Objective:",
        "Evaluate predictive performance and fairness behavior across gender subgroups for liver disease classification.",
        "",
        "End-to-end highlights:",
        "1. Dataset contains 583 samples with strong subgroup imbalance (male 75.64%, female 24.36%).",
        "2. Stage 2 identified subgroup distribution shift and outlier-heavy clinical features.",
        "3. Stage 3 applied fairness-aware preparation: 4-class stratified split, robust scaling, and clinically grounded feature engineering.",
        "4. Stage 4 tested three models across three experiment settings (global with/without gender, plus subgroup-specific models).",
        "5. Stage 5 confirmed consistent male-favored recall disparity in multiple settings.",
        "",
        "Best overall global model:",
        f"- Experiment: {best_global['experiment']}",
        f"- Model: {best_global['model']}",
        f"- Accuracy: {best_global['full_accuracy']:.4f}",
        f"- F1-score: {best_global['full_f1']:.4f}",
        f"- ROC-AUC: {best_global['full_roc_auc']:.4f}",
        "",
        "Most fairness-sensitive configuration (lowest female FNR and low recall gap):",
        f"- Experiment: {best_fair['experiment']}",
        f"- Model: {best_fair['model']}",
        f"- Female FNR: {best_fair['female_fnr']:.4f}",
        f"- Recall gap (male-female): {best_fair['recall_gap_male_minus_female']:+.4f}",
        "",
        "Executive decision:",
        "- Use the best global model as primary performance reference.",
        "- Include fairness-sensitive alternative model in discussion for healthcare deployment considerations.",
    ]
    return "\n".join(lines)


def _stage6_conclusion(df: pd.DataFrame) -> str:
    best_global = _best_global(df)

    exp1 = df[df["experiment"] == "exp1"].copy()
    min_gap_exp1 = exp1.iloc[exp1["recall_gap_male_minus_female"].abs().argmin()]

    lines = [
        "Stage 6 - Conclusion Draft",
        "",
        "This study investigated whether machine learning models for liver disease prediction behave differently across gender subgroups.",
        "Results from Stage 5 demonstrate that subgroup disparity is non-negligible and clinically relevant.",
        "",
        "Main conclusions:",
        f"1. The highest global predictive performance was achieved by {best_global['experiment']} / {best_global['model']} "
        f"(F1={best_global['full_f1']:.4f}, Accuracy={best_global['full_accuracy']:.4f}, AUC={best_global['full_roc_auc']:.4f}).",
        "2. Across multiple settings, male recall remained higher than female recall, indicating potential fairness risk.",
        f"3. In global-with-gender modeling (exp1), the smallest recall disparity was from {min_gap_exp1['model']} "
        f"(RecallGap={min_gap_exp1['recall_gap_male_minus_female']:+.4f}).",
        "4. Female false negative rate (FNR) varied substantially by model configuration, which has direct clinical implications for missed diagnosis risk.",
        "5. Removing explicit gender input did not eliminate disparity, suggesting subgroup differences are embedded in other clinical features.",
        "",
        "Fairness discussion:",
        "- In healthcare, recall and FNR are critical because false negatives can delay treatment.",
        "- The observed recall/FNR gaps suggest aggregate metrics alone are insufficient for safe model assessment.",
        "- Subgroup-specific reporting must be treated as mandatory in medical AI validation.",
        "",
        "Limitations:",
        "- Small dataset size may limit estimate stability.",
        "- Gender imbalance reduces robustness of female subgroup estimates.",
        "- Dataset is region-specific (India), limiting cross-region external validity.",
        "",
        "Overall, the results support the hypothesis that subgroup-aware evaluation provides materially better insight than aggregate-only evaluation.",
    ]
    return "\n".join(lines)


def _stage6_future_work() -> str:
    lines = [
        "Stage 6 - Future Work",
        "",
        "1) Data expansion",
        "- Increase sample size with more balanced female representation.",
        "- Collect multi-center and multi-hospital data to reduce sampling bias.",
        "",
        "2) External validation",
        "- Validate trained models on datasets from other regions and demographics.",
        "- Compare calibration and recall stability across populations.",
        "",
        "3) Fairness-aware modeling",
        "- Evaluate threshold-optimization per subgroup to reduce recall/FNR gap.",
        "- Test fairness-constrained objectives and post-processing approaches.",
        "",
        "4) Clinical utility enhancement",
        "- Incorporate decision-curve analysis and cost-sensitive evaluation aligned with clinical priorities.",
        "- Quantify acceptable false-positive tradeoff for reducing false negatives in female patients.",
        "",
        "5) Uncertainty reporting",
        "- Extend bootstrap confidence intervals to all key subgroup metrics.",
        "- Report interval overlap and stability under repeated split/seed sensitivity checks.",
    ]
    return "\n".join(lines)


def _master_summary(df: pd.DataFrame) -> str:
    best_global = _best_global(df)

    reports = sorted([p.name for p in REPORTS_DIR.glob("*.txt")])
    plots = sorted([p.name for p in PLOTS_DIR.glob("*.png")])

    lines = [
        "99 - Master Research Summary",
        "",
        "CRISP-DM execution summary:",
        "- Stage 1: Business understanding completed with formal problem framing, research questions, and hypotheses.",
        "- Stage 2: Data understanding completed with EDA reports, data profile tables, and visualization artifacts.",
        "- Stage 3: Data preparation completed with robust preprocessing and subgroup-aware split strategy.",
        "- Stage 4: Modeling completed with three algorithms across three experiment scenarios.",
        "- Stage 5: Evaluation completed with fairness gap analysis and clinical error interpretation.",
        "- Stage 6: Research packaging completed with executive summary, conclusion, and future-work agenda.",
        "",
        "Best model recommendation:",
        f"- {best_global['experiment']} / {best_global['model']} (F1={best_global['full_f1']:.4f}, Accuracy={best_global['full_accuracy']:.4f}, AUC={best_global['full_roc_auc']:.4f}).",
        "",
        "Fairness conclusion (critical insight):",
        "- Male-favored recall is consistently observed across major configurations.",
        "- Female false-negative risk is configuration-sensitive and must be explicitly monitored.",
        "- Subgroup-aware evaluation is necessary for healthcare-grade model validation.",
        "",
        "Generated plots list:",
        *[f"- {name}" for name in plots],
        "",
        "Generated reports list:",
        *[f"- {name}" for name in reports],
    ]
    return "\n".join(lines)


def run_stage_6() -> None:
    ensure_directories()

    model_comp_df = _load_stage5_table()

    (REPORTS_DIR / "06_executive_summary.txt").write_text(
        _stage6_executive_summary(model_comp_df), encoding="utf-8"
    )
    (REPORTS_DIR / "06_conclusion_draft.txt").write_text(
        _stage6_conclusion(model_comp_df), encoding="utf-8"
    )
    (REPORTS_DIR / "06_future_work.txt").write_text(
        _stage6_future_work(), encoding="utf-8"
    )
    (REPORTS_DIR / "99_master_research_summary.txt").write_text(
        _master_summary(model_comp_df), encoding="utf-8"
    )


if __name__ == "__main__":
    run_stage_6()
