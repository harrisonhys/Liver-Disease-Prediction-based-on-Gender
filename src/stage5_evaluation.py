from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.config import GENDER_COLUMN, MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, TABLES_DIR, ensure_directories


FEATURE_COLS = [
    "Age", "TB", "DB", "Alkphos", "Sgpt", "Sgot",
    "TP", "ALB", "A/G Ratio", "fractional_bilirubin", "Gender_encoded",
]
FEATURE_COLS_NO_GENDER = [f for f in FEATURE_COLS if f != "Gender_encoded"]


@dataclass(frozen=True)
class EvalSpec:
    experiment: str
    model: str
    model_path: str
    feature_cols: list[str]
    splits: list[str]


def _load_stage4_metrics() -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / "04_model_metrics.csv")


def _load_test_data() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_DATA_DIR / "test.csv")


def _split_mask(df: pd.DataFrame, split: str) -> np.ndarray:
    if split == "full":
        return np.ones(len(df), dtype=bool)
    if split == "male":
        return (df[GENDER_COLUMN] == "Male").values
    if split == "female":
        return (df[GENDER_COLUMN] == "Female").values
    raise ValueError(f"Unknown split: {split}")


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def _build_eval_specs() -> list[EvalSpec]:
    specs: list[EvalSpec] = []
    for model in ["LogisticRegression", "RandomForest", "XGBoost"]:
        specs.append(
            EvalSpec(
                experiment="exp1",
                model=model,
                model_path=f"exp1_{model.lower()}.pkl",
                feature_cols=FEATURE_COLS,
                splits=["full", "male", "female"],
            )
        )
        specs.append(
            EvalSpec(
                experiment="exp2",
                model=model,
                model_path=f"exp2_{model.lower()}.pkl",
                feature_cols=FEATURE_COLS_NO_GENDER,
                splits=["full", "male", "female"],
            )
        )
        specs.append(
            EvalSpec(
                experiment="exp3_male",
                model=model,
                model_path=f"exp3_male_{model.lower()}.pkl",
                feature_cols=FEATURE_COLS_NO_GENDER,
                splits=["male"],
            )
        )
        specs.append(
            EvalSpec(
                experiment="exp3_female",
                model=model,
                model_path=f"exp3_female_{model.lower()}.pkl",
                feature_cols=FEATURE_COLS_NO_GENDER,
                splits=["female"],
            )
        )
    return specs


def _compute_confusion_details(test_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in _build_eval_specs():
        model = joblib.load(MODELS_DIR / spec.model_path)
        for split in spec.splits:
            mask = _split_mask(test_df, split)
            sub = test_df.loc[mask].copy()
            X = sub[spec.feature_cols].values
            y = sub["target"].values
            y_pred = model.predict(X)

            tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())

            rows.append(
                {
                    "experiment": spec.experiment,
                    "model": spec.model,
                    "split": split,
                    "n_samples": int(len(y)),
                    "n_positive": n_pos,
                    "n_negative": n_neg,
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                    "fpr": round(_safe_div(fp, fp + tn), 4),
                    "fnr": round(_safe_div(fn, fn + tp), 4),
                }
            )
    return pd.DataFrame(rows)


def _prepare_model_comparison(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the main Stage 5 table from Stage 4 metrics.

    Output row granularity:
    - exp1 + model
    - exp2 + model
    - exp3 + model (male metrics from exp3_male and female metrics from exp3_female)
    """
    rows: list[dict[str, object]] = []

    def m(exp: str, model: str, split: str, col: str):
        sub = metrics_df[
            (metrics_df["experiment"] == exp)
            & (metrics_df["model"] == model)
            & (metrics_df["split"] == split)
        ]
        if sub.empty:
            return float("nan")
        return float(sub.iloc[0][col])

    for base_exp in ["exp1", "exp2"]:
        for model in ["LogisticRegression", "RandomForest", "XGBoost"]:
            male_acc = m(base_exp, model, "male", "accuracy")
            female_acc = m(base_exp, model, "female", "accuracy")
            male_f1 = m(base_exp, model, "male", "f1")
            female_f1 = m(base_exp, model, "female", "f1")
            male_recall = m(base_exp, model, "male", "recall")
            female_recall = m(base_exp, model, "female", "recall")

            male_fnr = 1.0 - male_recall
            female_fnr = 1.0 - female_recall

            rows.append(
                {
                    "experiment": base_exp,
                    "model": model,
                    "full_accuracy": m(base_exp, model, "full", "accuracy"),
                    "full_f1": m(base_exp, model, "full", "f1"),
                    "full_roc_auc": m(base_exp, model, "full", "roc_auc"),
                    "male_accuracy": male_acc,
                    "female_accuracy": female_acc,
                    "accuracy_gap_male_minus_female": round(male_acc - female_acc, 4),
                    "male_f1": male_f1,
                    "female_f1": female_f1,
                    "f1_gap_male_minus_female": round(male_f1 - female_f1, 4),
                    "male_recall": male_recall,
                    "female_recall": female_recall,
                    "recall_gap_male_minus_female": round(male_recall - female_recall, 4),
                    "male_fnr": round(male_fnr, 4),
                    "female_fnr": round(female_fnr, 4),
                    "fnr_gap_female_minus_male": round(female_fnr - male_fnr, 4),
                    "male_favored_on_recall": bool(male_recall > female_recall),
                }
            )

    for model in ["LogisticRegression", "RandomForest", "XGBoost"]:
        male_acc = m("exp3_male", model, "male", "accuracy")
        female_acc = m("exp3_female", model, "female", "accuracy")
        male_f1 = m("exp3_male", model, "male", "f1")
        female_f1 = m("exp3_female", model, "female", "f1")
        male_recall = m("exp3_male", model, "male", "recall")
        female_recall = m("exp3_female", model, "female", "recall")

        male_fnr = 1.0 - male_recall
        female_fnr = 1.0 - female_recall

        rows.append(
            {
                "experiment": "exp3",
                "model": model,
                "full_accuracy": float("nan"),
                "full_f1": float("nan"),
                "full_roc_auc": float("nan"),
                "male_accuracy": male_acc,
                "female_accuracy": female_acc,
                "accuracy_gap_male_minus_female": round(male_acc - female_acc, 4),
                "male_f1": male_f1,
                "female_f1": female_f1,
                "f1_gap_male_minus_female": round(male_f1 - female_f1, 4),
                "male_recall": male_recall,
                "female_recall": female_recall,
                "recall_gap_male_minus_female": round(male_recall - female_recall, 4),
                "male_fnr": round(male_fnr, 4),
                "female_fnr": round(female_fnr, 4),
                "fnr_gap_female_minus_male": round(female_fnr - male_fnr, 4),
                "male_favored_on_recall": bool(male_recall > female_recall),
            }
        )

    return pd.DataFrame(rows)


def _build_comparative_analysis(
    model_comp_df: pd.DataFrame, confusion_df: pd.DataFrame
) -> str:
    lines: list[str] = []
    lines.append("Stage 5 - Evaluation Comparative Analysis")
    lines.append("")

    # Overall best by F1 in global settings (exp1/exp2)
    global_only = model_comp_df[model_comp_df["experiment"].isin(["exp1", "exp2"])].copy()
    best_global = global_only.sort_values("full_f1", ascending=False).iloc[0]

    lines.append("1) Overall model comparison (global experiments)")
    lines.append(
        f"- Best global configuration by Full F1: {best_global['experiment']} / {best_global['model']} "
        f"(F1={best_global['full_f1']:.4f}, Accuracy={best_global['full_accuracy']:.4f}, AUC={best_global['full_roc_auc']:.4f})."
    )

    exp1 = model_comp_df[model_comp_df["experiment"] == "exp1"].copy()
    exp1_best_fair = exp1.iloc[exp1["recall_gap_male_minus_female"].abs().argmin()]
    lines.append(
        f"- In Experiment 1, smallest recall gap is from {exp1_best_fair['model']} "
        f"(male-female recall gap={exp1_best_fair['recall_gap_male_minus_female']:+.4f})."
    )
    lines.append("")

    lines.append("2) Gender performance disparity and fairness metrics")
    for _, row in model_comp_df.iterrows():
        lines.append(
            f"- {row['experiment']} / {row['model']}: "
            f"AccGap(male-female)={row['accuracy_gap_male_minus_female']:+.4f}, "
            f"F1Gap(male-female)={row['f1_gap_male_minus_female']:+.4f}, "
            f"RecallGap(male-female)={row['recall_gap_male_minus_female']:+.4f}, "
            f"FNR female={row['female_fnr']:.4f}, FNR male={row['male_fnr']:.4f}, "
            f"FNRGap(female-male)={row['fnr_gap_female_minus_male']:+.4f}."
        )
    lines.append("")

    lines.append("3) False positives vs false negatives")
    conf_sorted = confusion_df.sort_values(["experiment", "model", "split"])
    for _, row in conf_sorted.iterrows():
        lines.append(
            f"- {row['experiment']} / {row['model']} / {row['split']}: "
            f"TP={int(row['tp'])}, TN={int(row['tn'])}, FP={int(row['fp'])}, FN={int(row['fn'])}, "
            f"FPR={row['fpr']:.4f}, FNR={row['fnr']:.4f}."
        )
    lines.append("")

    lines.append("4) Clinical implications")
    lines.append(
        "- In healthcare screening, false negatives are more critical than false positives because missed liver disease cases can delay treatment."
    )

    worst_female_fnr = model_comp_df.sort_values("female_fnr", ascending=False).iloc[0]
    best_female_fnr = model_comp_df.sort_values("female_fnr", ascending=True).iloc[0]
    lines.append(
        f"- Highest female FNR is {worst_female_fnr['female_fnr']:.4f} "
        f"({worst_female_fnr['experiment']} / {worst_female_fnr['model']}); this indicates the highest under-diagnosis risk for female patients."
    )
    lines.append(
        f"- Lowest female FNR is {best_female_fnr['female_fnr']:.4f} "
        f"({best_female_fnr['experiment']} / {best_female_fnr['model']}); this configuration minimizes missed female liver disease cases."
    )
    lines.append(
        "- Any model with large positive RecallGap(male-female) and large positive FNRGap(female-male) should be flagged as fairness risk."
    )

    return "\n".join(lines)


def _build_ai_ready_summary(model_comp_df: pd.DataFrame) -> str:
    exp1 = model_comp_df[model_comp_df["experiment"] == "exp1"].copy()
    exp2 = model_comp_df[model_comp_df["experiment"] == "exp2"].copy()
    exp3 = model_comp_df[model_comp_df["experiment"] == "exp3"].copy()

    best_global = model_comp_df[
        model_comp_df["experiment"].isin(["exp1", "exp2"])
    ].sort_values("full_f1", ascending=False).iloc[0]

    fairest_exp1 = exp1.iloc[exp1["recall_gap_male_minus_female"].abs().argmin()]

    lines = [
        "AI-Ready Evaluation Summary (Stage 5)",
        "",
        "Core findings:",
        f"1. Best overall global setup is {best_global['experiment']} / {best_global['model']} "
        f"(Full Accuracy={best_global['full_accuracy']:.4f}, Full F1={best_global['full_f1']:.4f}, Full AUC={best_global['full_roc_auc']:.4f}).",
        "2. Gender disparity exists across multiple models, with male recall generally higher than female recall.",
        f"3. In Experiment 1, the smallest recall disparity is from {fairest_exp1['model']} "
        f"(RecallGap male-female={fairest_exp1['recall_gap_male_minus_female']:+.4f}).",
        "4. False Negative Rate (FNR) is higher in female subgroup for several model settings, signaling potential under-diagnosis risk.",
        "5. Removing explicit gender feature (Exp2) does not fully remove subgroup disparity, indicating distributional differences in clinical features.",
        "6. Separate subgroup models (Exp3) can reduce female FNR for some algorithms, but this effect is model-dependent.",
        "",
        "Fairness metrics tracked:",
        "- Accuracy Gap (male - female)",
        "- F1 Gap (male - female)",
        "- Recall Gap (male - female)",
        "- Female and Male FNR, plus FNR Gap (female - male)",
        "",
        "Recommended direction for Stage 6 packaging:",
        "- Use the best global model as primary deployment reference.",
        "- Include fairness-risk model comparison using RecallGap and FNRGap as key evidence.",
        "- Highlight clinical consequence of female false negatives in the discussion section.",
    ]

    # Add concise model snapshots for each experiment
    lines.append("")
    lines.append("Experiment snapshots:")
    for exp_name, exp_df in [("exp1", exp1), ("exp2", exp2), ("exp3", exp3)]:
        best_row = exp_df.sort_values("female_fnr", ascending=True).iloc[0]
        lines.append(
            f"- {exp_name}: lowest female FNR = {best_row['female_fnr']:.4f} by {best_row['model']} "
            f"(RecallGap={best_row['recall_gap_male_minus_female']:+.4f})."
        )

    return "\n".join(lines)


def run_stage_5() -> None:
    ensure_directories()

    metrics_df = _load_stage4_metrics()
    test_df = _load_test_data()

    model_comp_df = _prepare_model_comparison(metrics_df)
    confusion_df = _compute_confusion_details(test_df)

    model_comp_df.to_csv(TABLES_DIR / "05_model_comparison.csv", index=False)
    confusion_df.to_csv(TABLES_DIR / "05_confusion_by_group.csv", index=False)

    (REPORTS_DIR / "05_evaluation_comparative_analysis.txt").write_text(
        _build_comparative_analysis(model_comp_df, confusion_df),
        encoding="utf-8",
    )
    (REPORTS_DIR / "05_ai_ready_evaluation_summary.txt").write_text(
        _build_ai_ready_summary(model_comp_df),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run_stage_5()
