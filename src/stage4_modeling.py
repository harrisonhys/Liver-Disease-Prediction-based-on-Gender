from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

from src.config import (
    GENDER_COLUMN,
    MODELS_DIR,
    PLOTS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    ensure_directories,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
BOOTSTRAP_N = 1000
_rng = np.random.default_rng(RANDOM_STATE)

FEATURE_COLS = [
    "Age", "TB", "DB", "Alkphos", "Sgpt", "Sgot",
    "TP", "ALB", "A/G Ratio", "fractional_bilirubin", "Gender_encoded",
]
FEATURE_COLS_NO_GENDER = [f for f in FEATURE_COLS if f != "Gender_encoded"]

SPLIT_LABELS = {
    "full": "Full Test Set",
    "male": "Male Subgroup",
    "female": "Female Subgroup",
}
MODEL_DISPLAY = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
}
MODEL_COLORS = {
    "LogisticRegression": "#e41a1c",
    "RandomForest": "#377eb8",
    "XGBoost": "#4daf4a",
}


# ─────────────────────────────── Data helpers ───────────────────────────────

def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    return train, test


def _make_models() -> dict:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, random_state=RANDOM_STATE,
            eval_metric="logloss", verbosity=0,
        ),
    }


def _split_by_gender(
    df: pd.DataFrame, feature_cols: list[str]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    masks = {
        "full": np.ones(len(df), dtype=bool),
        "male": (df[GENDER_COLUMN] == "Male").values,
        "female": (df[GENDER_COLUMN] == "Female").values,
    }
    X_all = df[feature_cols].values
    y_all = df["target"].values
    return {k: (X_all[m], y_all[m]) for k, m in masks.items()}


# ──────────────────────────────── Metrics ───────────────────────────────────

def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
) -> dict:
    safe = len(np.unique(y_true)) > 1
    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_true, y_prob)), 4) if safe else float("nan"),
    }


def bootstrap_recall_ci(
    y_true: np.ndarray, y_prob: np.ndarray, n: int = BOOTSTRAP_N
) -> tuple[float, float]:
    """95% bootstrap CI for recall (for small female subgroup)."""
    recalls = []
    for _ in range(n):
        idx = _rng.integers(0, len(y_true), len(y_true))
        yt = y_true[idx]
        yp = (y_prob[idx] >= 0.5).astype(int)
        if yt.sum() > 0:
            recalls.append(float(recall_score(yt, yp, zero_division=0)))
    if len(recalls) < 10:
        return float("nan"), float("nan")
    return (
        round(float(np.percentile(recalls, 2.5)), 4),
        round(float(np.percentile(recalls, 97.5)), 4),
    )


def evaluate_splits(
    model: object,
    test: pd.DataFrame,
    feature_cols: list[str],
    experiment: str,
    model_name: str,
) -> list[dict]:
    rows = []
    for split, (X, y) in _split_by_gender(test, feature_cols).items():
        if len(y) == 0:
            continue
        y_pred = model.predict(X)  # type: ignore[attr-defined]
        y_prob = model.predict_proba(X)[:, 1]  # type: ignore[attr-defined]
        metrics = compute_metrics(y, y_pred, y_prob)
        ci_lo, ci_hi = (
            bootstrap_recall_ci(y, y_prob) if split == "female" else (None, None)
        )
        rows.append(
            {
                "experiment": experiment,
                "model": model_name,
                "split": split,
                "n_samples": int(len(y)),
                **metrics,
                "recall_ci_lower": ci_lo,
                "recall_ci_upper": ci_hi,
            }
        )
    return rows


# ──────────────────────────── Plotting helpers ──────────────────────────────

def _savefig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cm_panel(
    model: object,
    test: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    exp_slug: str,
) -> None:
    """One 1×3 figure — confusion matrices for full / male / female splits."""
    splits = _split_by_gender(test, feature_cols)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, split in zip(axes, ["full", "male", "female"]):
        X, y = splits[split]
        y_pred = model.predict(X)  # type: ignore[attr-defined]
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No LD", "LD"], yticklabels=["No LD", "LD"], cbar=False,
        )
        ax.set_title(f"{SPLIT_LABELS[split]} (n={len(y)})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle(
        f"Confusion Matrices — {MODEL_DISPLAY[model_name]} ({exp_slug})", fontsize=13
    )
    _savefig(fig, PLOTS_DIR / f"04_{exp_slug}_{model_name.lower()[:2]}_cm.png")


def plot_roc(
    trained: list[tuple[str, object]],
    test: pd.DataFrame,
    feature_cols: list[str],
    exp_slug: str,
) -> None:
    """1×3 ROC figure — all models per panel, split by [full / male / female]."""
    splits = _split_by_gender(test, feature_cols)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, split in zip(axes, ["full", "male", "female"]):
        X, y = splits[split]
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        if len(np.unique(y)) < 2:
            ax.set_title(f"{SPLIT_LABELS[split]} — single class")
            continue
        for mn, model in trained:
            yp = model.predict_proba(X)[:, 1]  # type: ignore[attr-defined]
            fpr, tpr, _ = roc_curve(y, yp)
            ax.plot(
                fpr, tpr, color=MODEL_COLORS[mn], lw=1.5,
                label=f"{MODEL_DISPLAY[mn]} ({roc_auc_score(y, yp):.3f})",
            )
        ax.set(
            title=f"{SPLIT_LABELS[split]} (n={len(y)})",
            xlabel="FPR", ylabel="TPR", xlim=[0, 1], ylim=[0, 1.02],
        )
        ax.legend(fontsize=7)
    fig.suptitle(f"ROC Curves ({exp_slug})", fontsize=13)
    _savefig(fig, PLOTS_DIR / f"04_{exp_slug}_roc.png")


def plot_pr(
    trained: list[tuple[str, object]],
    test: pd.DataFrame,
    feature_cols: list[str],
    exp_slug: str,
) -> None:
    """1×3 PR figure — all models per panel."""
    splits = _split_by_gender(test, feature_cols)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, split in zip(axes, ["full", "male", "female"]):
        X, y = splits[split]
        if len(np.unique(y)) < 2:
            ax.set_title(f"{SPLIT_LABELS[split]} — single class")
            continue
        for mn, model in trained:
            yp = model.predict_proba(X)[:, 1]  # type: ignore[attr-defined]
            prec, rec, _ = precision_recall_curve(y, yp)
            ap = average_precision_score(y, yp)
            ax.plot(
                rec, prec, color=MODEL_COLORS[mn], lw=1.5,
                label=f"{MODEL_DISPLAY[mn]} (AP={ap:.3f})",
            )
        baseline = float(y.mean())
        ax.axhline(baseline, color="gray", ls="--", lw=0.8, label=f"Baseline ({baseline:.2f})")
        ax.set(
            title=f"{SPLIT_LABELS[split]} (n={len(y)})",
            xlabel="Recall", ylabel="Precision", xlim=[0, 1], ylim=[0, 1.05],
        )
        ax.legend(fontsize=7)
    fig.suptitle(f"Precision-Recall Curves ({exp_slug})", fontsize=13)
    _savefig(fig, PLOTS_DIR / f"04_{exp_slug}_pr.png")


def plot_feature_importance(
    model: object,
    feature_cols: list[str],
    model_name: str,
    exp_slug: str,
) -> None:
    if hasattr(model, "coef_"):
        imp = np.abs(model.coef_[0])  # type: ignore[attr-defined]
    elif hasattr(model, "feature_importances_"):
        imp = model.feature_importances_  # type: ignore[attr-defined]
    else:
        return
    imp_df = (
        pd.DataFrame({"feature": feature_cols, "importance": imp})
        .sort_values("importance")
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(imp_df)))
    ax.barh(imp_df["feature"], imp_df["importance"], color=colors)
    ax.set_title(f"Feature Importance — {MODEL_DISPLAY[model_name]} ({exp_slug})")
    ax.set_xlabel("Importance")
    _savefig(fig, PLOTS_DIR / f"04_{exp_slug}_{model_name.lower()[:2]}_importance.png")


def plot_shap(
    model: object,
    X_train_df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    exp_slug: str,
) -> None:
    try:
        n_bg = min(200, len(X_train_df))
        X_bg_df = X_train_df[feature_cols].iloc[:n_bg].copy()
        X_bg = X_bg_df.values

        if model_name == "LogisticRegression":
            explainer = shap.LinearExplainer(model, X_bg)
            sv = explainer.shap_values(X_bg)
        else:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_bg)
            if isinstance(sv, list):
                sv = sv[1]
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                sv = sv[:, :, 1]

        shap.summary_plot(sv, X_bg_df, show=False, plot_size=None)
        plt.gcf().suptitle(
            f"SHAP Summary — {MODEL_DISPLAY[model_name]} ({exp_slug})",
            fontsize=11, y=1.01,
        )
        plt.tight_layout()
        plt.savefig(
            PLOTS_DIR / f"04_{exp_slug}_{model_name.lower()[:2]}_shap.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close("all")
    except Exception as exc:
        plt.close("all")
        print(f"  [SHAP skipped] {model_name} ({exp_slug}): {exc}")


# ──────────────────────────────── Experiments ───────────────────────────────

def _run_global_experiment(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    exp_slug: str,
    save_shap: bool,
) -> list[dict]:
    """Generic runner for Exp1 (with gender) and Exp2 (without gender)."""
    X_train = train[feature_cols].values
    y_train = train["target"].values
    models = _make_models()
    all_metrics: list[dict] = []
    trained: list[tuple[str, object]] = []

    for mn, model in models.items():
        print(f"  Training {MODEL_DISPLAY[mn]} ({exp_slug})…")
        model.fit(X_train, y_train)
        joblib.dump(model, MODELS_DIR / f"{exp_slug}_{mn.lower()}.pkl")

        all_metrics.extend(evaluate_splits(model, test, feature_cols, exp_slug, mn))
        trained.append((mn, model))

        plot_cm_panel(model, test, feature_cols, mn, exp_slug)
        plot_feature_importance(model, feature_cols, mn, exp_slug)
        if save_shap:
            plot_shap(model, train, feature_cols, mn, exp_slug)

    plot_roc(trained, test, feature_cols, exp_slug)
    plot_pr(trained, test, feature_cols, exp_slug)
    return all_metrics


def run_experiment_1(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    """Experiment 1 — Global model WITH Gender feature."""
    print("\n[Experiment 1] Global model with Gender feature")
    return _run_global_experiment(train, test, FEATURE_COLS, "exp1", save_shap=True)


def run_experiment_2(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    """Experiment 2 — Global model WITHOUT Gender feature."""
    print("\n[Experiment 2] Global model WITHOUT Gender feature")
    return _run_global_experiment(
        train, test, FEATURE_COLS_NO_GENDER, "exp2", save_shap=False
    )


def run_experiment_3(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    """Experiment 3 — Separate models per gender; no Gender feature in inputs."""
    print("\n[Experiment 3] Separate gender-specific models")
    feature_cols = FEATURE_COLS_NO_GENDER
    all_metrics: list[dict] = []

    for gender_key, gender_val in [("male", "Male"), ("female", "Female")]:
        exp_slug = f"exp3_{gender_key}"
        train_g = train[train[GENDER_COLUMN] == gender_val].copy()
        test_g = test[test[GENDER_COLUMN] == gender_val].copy()
        X_train_g = train_g[feature_cols].values
        y_train_g = train_g["target"].values

        if len(np.unique(y_train_g)) < 2:
            print(f"  Skipping {gender_val} — single class in training data")
            continue

        models = _make_models()
        trained: list[tuple[str, object]] = []

        for mn, model in models.items():
            print(f"  Training {MODEL_DISPLAY[mn]} ({exp_slug})…")
            model.fit(X_train_g, y_train_g)
            joblib.dump(model, MODELS_DIR / f"{exp_slug}_{mn.lower()}.pkl")

            X_g = test_g[feature_cols].values
            y_g = test_g["target"].values
            y_pred = model.predict(X_g)
            y_prob = model.predict_proba(X_g)[:, 1]
            metrics = compute_metrics(y_g, y_pred, y_prob)
            ci_lo, ci_hi = (
                bootstrap_recall_ci(y_g, y_prob) if gender_key == "female" else (None, None)
            )
            all_metrics.append(
                {
                    "experiment": exp_slug,
                    "model": mn,
                    "split": gender_key,
                    "n_samples": int(len(y_g)),
                    **metrics,
                    "recall_ci_lower": ci_lo,
                    "recall_ci_upper": ci_hi,
                }
            )
            trained.append((mn, model))

            # Per-model single confusion matrix
            cm = confusion_matrix(y_g, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No LD", "LD"], yticklabels=["No LD", "LD"], cbar=False,
            )
            ax.set_title(
                f"CM — {MODEL_DISPLAY[mn]}\n{gender_key.capitalize()} Model (n={len(y_g)})"
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            _savefig(fig, PLOTS_DIR / f"04_{exp_slug}_{mn.lower()[:2]}_cm.png")

            plot_feature_importance(model, feature_cols, mn, exp_slug)
            plot_shap(model, train_g, feature_cols, mn, exp_slug)

        # Combined ROC + PR in a single 1×2 figure for this gender
        X_g = test_g[feature_cols].values
        y_g = test_g["target"].values
        if len(np.unique(y_g)) > 1 and trained:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for mn_t, mdl in trained:
                yp = mdl.predict_proba(X_g)[:, 1]
                fpr, tpr, _ = roc_curve(y_g, yp)
                axes[0].plot(
                    fpr, tpr, color=MODEL_COLORS[mn_t], lw=1.5,
                    label=f"{MODEL_DISPLAY[mn_t]} ({roc_auc_score(y_g, yp):.3f})",
                )
                prec, rec, _ = precision_recall_curve(y_g, yp)
                ap = average_precision_score(y_g, yp)
                axes[1].plot(
                    rec, prec, color=MODEL_COLORS[mn_t], lw=1.5,
                    label=f"{MODEL_DISPLAY[mn_t]} (AP={ap:.3f})",
                )
            axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
            axes[0].set(
                title=f"ROC — {gender_key.capitalize()} Model (n={len(y_g)})",
                xlabel="FPR", ylabel="TPR", xlim=[0, 1], ylim=[0, 1.02],
            )
            axes[0].legend(fontsize=8)
            axes[1].axhline(float(y_g.mean()), color="gray", ls="--", lw=0.8)
            axes[1].set(
                title=f"PR — {gender_key.capitalize()} Model (n={len(y_g)})",
                xlabel="Recall", ylabel="Precision", xlim=[0, 1], ylim=[0, 1.05],
            )
            axes[1].legend(fontsize=8)
            fig.suptitle(
                f"Experiment 3 — {gender_key.capitalize()} Subgroup Models", fontsize=13
            )
            _savefig(fig, PLOTS_DIR / f"04_{exp_slug}_roc_pr.png")

    return all_metrics


# ─────────────────────────────── Reports ────────────────────────────────────

def _fmt_ci(lo, hi) -> str:
    if lo is None or (isinstance(lo, float) and np.isnan(lo)):
        return "     —     "
    return f"[{lo:.3f},{hi:.3f}]"


def build_exp_report(
    metrics_df: pd.DataFrame, exp_prefix: str, description: str
) -> str:
    sub = metrics_df[metrics_df["experiment"].str.startswith(exp_prefix)]
    header = [
        f"Stage 4 — Modeling Report: {exp_prefix}",
        f"Description: {description}",
        "",
    ]
    body: list[str] = []
    for mn in ["LogisticRegression", "RandomForest", "XGBoost"]:
        rows = sub[sub["model"] == mn]
        if rows.empty:
            continue
        body.append(f"\n{MODEL_DISPLAY[mn]}:")
        body.append(
            f"{'Split':<10} {'N':>5} {'Acc':>7} {'Prec':>7} {'Rec':>7} "
            f"{'F1':>7} {'AUC':>7}  {'Recall 95% CI':>14}"
        )
        body.append("-" * 75)
        for _, row in rows.iterrows():
            body.append(
                f"{str(row['split']):<10} {int(row['n_samples']):>5} "
                f"{row['accuracy']:>7.4f} {row['precision']:>7.4f} "
                f"{row['recall']:>7.4f} {row['f1']:>7.4f} {row['roc_auc']:>7.4f}  "
                f"{_fmt_ci(row['recall_ci_lower'], row['recall_ci_upper']):>14}"
            )
    return "\n".join(header + body)


def build_ai_ready_summary(metrics_df: pd.DataFrame) -> str:
    exp1 = metrics_df[metrics_df["experiment"] == "exp1"]

    best = (
        exp1[exp1["split"] == "full"]
        .sort_values("f1", ascending=False)
        .iloc[0]
    )

    male_rec = (
        exp1[exp1["split"] == "male"][["model", "recall"]]
        .set_index("model")["recall"]
    )
    female_rec = (
        exp1[exp1["split"] == "female"][["model", "recall"]]
        .set_index("model")["recall"]
    )
    gaps = (male_rec - female_rec).round(4)

    exp3_female = metrics_df[metrics_df["experiment"] == "exp3_female"]
    exp1_female = exp1[exp1["split"] == "female"][["model", "recall"]].set_index("model")

    lines = [
        "AI-Ready Modeling Summary (Stage 4)",
        "",
        f"Best model overall (Exp1, full test): {MODEL_DISPLAY.get(str(best['model']), best['model'])}",
        f"  Accuracy={best['accuracy']:.4f}  F1={best['f1']:.4f}  AUC={best['roc_auc']:.4f}",
        "",
        "Recall gap (Male − Female) by model — Experiment 1:",
    ]
    for mn, gap in gaps.items():
        direction = "male favored" if gap > 0.0 else "female favored" if gap < 0.0 else "equal"
        lines.append(f"  {MODEL_DISPLAY.get(mn, mn):24s}  gap = {gap:+.4f}  ({direction})")

    lines += [
        "",
        "Experiment 1 vs Experiment 2 (effect of removing Gender feature):",
    ]
    exp2_female = metrics_df[
        (metrics_df["experiment"] == "exp2") & (metrics_df["split"] == "female")
    ][["model", "recall"]].set_index("model")
    for mn in ["LogisticRegression", "RandomForest", "XGBoost"]:
        r1 = exp1_female.loc[mn, "recall"] if mn in exp1_female.index else float("nan")
        r2 = exp2_female.loc[mn, "recall"] if mn in exp2_female.index else float("nan")
        delta = round(r2 - r1, 4)
        lines.append(
            f"  {MODEL_DISPLAY.get(mn):24s}  female recall Exp1={r1:.4f}  "
            f"Exp2={r2:.4f}  Δ={delta:+.4f}"
        )

    lines += [
        "",
        "Experiment 3 (separate models) vs Experiment 1 — female recall improvement:",
    ]
    for mn in ["LogisticRegression", "RandomForest", "XGBoost"]:
        r1 = exp1_female.loc[mn, "recall"] if mn in exp1_female.index else float("nan")
        r3_row = exp3_female[exp3_female["model"] == mn]
        r3 = float(r3_row["recall"].values[0]) if not r3_row.empty else float("nan")
        delta = round(r3 - r1, 4)
        ci_lo = r3_row["recall_ci_lower"].values[0] if not r3_row.empty else float("nan")
        ci_hi = r3_row["recall_ci_upper"].values[0] if not r3_row.empty else float("nan")
        ci_str = _fmt_ci(ci_lo, ci_hi)
        lines.append(
            f"  {MODEL_DISPLAY.get(mn):24s}  Exp1={r1:.4f}  "
            f"Exp3={r3:.4f}  Δ={delta:+.4f}  CI={ci_str}"
        )

    lines += [
        "",
        "Notes for Stage 5 (Evaluation):",
        "- All female-subgroup recall values include 95% bootstrap CI (n_female_test=28).",
        "- Use Δ recall (male−female) and Δ F1 as primary fairness gap metrics.",
        "- Compare Exp2 vs Exp1 to quantify how much Gender feature contributes to disparity.",
        "- Compare Exp3 vs Exp1 to assess whether gender-specific training closes the gap.",
    ]
    return "\n".join(lines)


# ──────────────────────────────── Pipeline ──────────────────────────────────

def run_stage_4() -> None:
    ensure_directories()
    train, test = load_processed_data()

    all_metrics: list[dict] = []
    all_metrics.extend(run_experiment_1(train, test))
    all_metrics.extend(run_experiment_2(train, test))
    all_metrics.extend(run_experiment_3(train, test))

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(TABLES_DIR / "04_model_metrics.csv", index=False)

    exp_descriptions = {
        "exp1": "Global model WITH Gender feature — evaluated on full / male / female test splits",
        "exp2": "Global model WITHOUT Gender feature — evaluated on full / male / female test splits",
        "exp3": "Separate models trained per gender subgroup (no Gender feature in inputs)",
    }
    for slug, desc in exp_descriptions.items():
        (REPORTS_DIR / f"04_modeling_{slug}.txt").write_text(
            build_exp_report(metrics_df, slug, desc), encoding="utf-8"
        )

    (REPORTS_DIR / "04_ai_ready_modeling_summary.txt").write_text(
        build_ai_ready_summary(metrics_df), encoding="utf-8"
    )
    print("\n[Stage 4 complete] Metrics, plots, models, and reports saved.")


if __name__ == "__main__":
    run_stage_4()
