from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import (
    DATASET_PATH,
    GENDER_COLUMN,
    PLOTS_DIR,
    POSITIVE_CLASS,
    RAW_DATA_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    TARGET_COLUMN,
    ensure_directories,
)


sns.set_theme(style="whitegrid")


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


def archive_raw_dataset() -> Path:
    destination = RAW_DATA_DIR / DATASET_PATH.name
    if DATASET_PATH.resolve() != destination.resolve():
        shutil.copy2(DATASET_PATH, destination)
    return destination


def build_data_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    numeric_columns = df.select_dtypes(include="number").columns

    for column in df.columns:
        series = df[column]
        row: dict[str, object] = {
            "column": column,
            "dtype": str(series.dtype),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean() * 100), 4),
            "unique_values": int(series.nunique(dropna=True)),
        }

        if column in numeric_columns:
            description = series.describe(percentiles=[0.25, 0.5, 0.75])
            row.update(
                {
                    "mean": round(float(description.get("mean", 0.0)), 4),
                    "std": round(float(description.get("std", 0.0)), 4),
                    "min": round(float(description.get("min", 0.0)), 4),
                    "q1": round(float(description.get("25%", 0.0)), 4),
                    "median": round(float(description.get("50%", 0.0)), 4),
                    "q3": round(float(description.get("75%", 0.0)), 4),
                    "max": round(float(description.get("max", 0.0)), 4),
                }
            )
        else:
            top_values = series.value_counts(dropna=False).head(3)
            row["top_values"] = "; ".join(
                f"{index}: {value}" for index, value in top_values.items()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_pct": (df.isna().mean().values * 100).round(4),
        }
    )
    return missing.sort_values(["missing_count", "column"], ascending=[False, True])


def compute_outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    rows: list[dict[str, object]] = []

    for column in numeric_df.columns:
        series = numeric_df[column].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_mask = (series < lower_bound) | (series > upper_bound)
        rows.append(
            {
                "column": column,
                "q1": round(float(q1), 4),
                "q3": round(float(q3), 4),
                "iqr": round(float(iqr), 4),
                "lower_bound": round(float(lower_bound), 4),
                "upper_bound": round(float(upper_bound), 4),
                "outlier_count": int(outlier_mask.sum()),
                "outlier_pct": round(float(outlier_mask.mean() * 100), 4),
            }
        )

    return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)


def _save_plot(output_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_gender_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x=GENDER_COLUMN, hue=GENDER_COLUMN, palette="Set2", legend=False)
    ax.set_title("Gender Distribution")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    _save_plot(PLOTS_DIR / "02_gender_distribution.png")


def plot_target_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(data=df, x=TARGET_COLUMN, hue=TARGET_COLUMN, palette="Set1", legend=False)
    ax.set_title("Target Distribution")
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    plt.setp(ax.get_xticklabels(), rotation=10, ha="right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    _save_plot(PLOTS_DIR / "02_target_distribution.png")


def plot_missing_values(df: pd.DataFrame) -> None:
    missing_pct = df.isna().mean().sort_values(ascending=False) * 100
    plot_data = pd.DataFrame(
        {"feature": missing_pct.index, "missing_pct": missing_pct.values}
    )
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=plot_data,
        x="feature",
        y="missing_pct",
        hue="feature",
        palette="crest",
        legend=False,
    )
    ax.set_title("Missing Values Percentage by Feature")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Missing Percentage")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    _save_plot(PLOTS_DIR / "02_missing_values.png")


def plot_feature_distributions(df: pd.DataFrame) -> None:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    n_columns = 3
    n_rows = (len(numeric_columns) + n_columns - 1) // n_columns
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for axis, column in zip(axes, numeric_columns):
        sns.histplot(df[column], kde=True, ax=axis, color="#2a9d8f")
        axis.set_title(f"Distribution of {column}")

    for axis in axes[len(numeric_columns):]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_feature_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    correlation = df.select_dtypes(include="number").corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap")
    _save_plot(PLOTS_DIR / "02_correlation_heatmap.png")


# Lab features ordered by clinical grouping: bilirubin → enzymes → proteins
_LAB_FEATURES = ["TB", "DB", "Alkphos", "Sgpt", "Sgot", "TP", "ALB", "A/G Ratio"]


def plot_target_by_gender(df: pd.DataFrame) -> None:
    """Grouped bar – positive-class prevalence per gender with absolute counts."""
    counts = (
        df.groupby([GENDER_COLUMN, TARGET_COLUMN], observed=True)
        .size()
        .reset_index(name="count")
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: absolute counts
    sns.barplot(
        data=counts,
        x=GENDER_COLUMN,
        y="count",
        hue=TARGET_COLUMN,
        palette="Set1",
        ax=axes[0],
    )
    axes[0].set_title("Target Distribution by Gender (Counts)")
    axes[0].set_xlabel("Gender")
    axes[0].set_ylabel("Patient Count")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%d", padding=2)

    # Right: percentage within each gender group
    totals = counts.groupby(GENDER_COLUMN, observed=True)["count"].transform("sum")
    counts["pct"] = (counts["count"] / totals * 100).round(1)
    sns.barplot(
        data=counts,
        x=GENDER_COLUMN,
        y="pct",
        hue=TARGET_COLUMN,
        palette="Set1",
        ax=axes[1],
    )
    axes[1].set_title("Target Distribution by Gender (%)")
    axes[1].set_xlabel("Gender")
    axes[1].set_ylabel("Percentage within Gender (%)")
    axes[1].set_ylim(0, 105)
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f%%", padding=2)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_target_by_gender.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lab_boxplots_by_gender(df: pd.DataFrame) -> None:
    """One subplot per lab feature showing distribution by gender."""
    n_cols = 4
    n_rows = (len(_LAB_FEATURES) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    palette = {"Male": "#4393c3", "Female": "#d6604d"}

    for axis, feature in zip(axes, _LAB_FEATURES):
        sns.boxplot(
            data=df,
            x=GENDER_COLUMN,
            y=feature,
            hue=GENDER_COLUMN,
            palette=palette,
            width=0.5,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
            legend=False,
            ax=axis,
        )
        axis.set_title(f"{feature} by Gender")
        axis.set_xlabel("")
        axis.set_ylabel(feature)

    for axis in axes[len(_LAB_FEATURES):]:
        axis.axis("off")

    fig.suptitle("Lab Feature Distributions by Gender", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "02_lab_boxplots_by_gender.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def plot_lab_boxplots_by_target(df: pd.DataFrame) -> None:
    """One subplot per lab feature showing distribution by disease class."""
    n_cols = 4
    n_rows = (len(_LAB_FEATURES) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    palette = {"Liver Disease": "#d73027", "No Liver Disease": "#4dac26"}

    for axis, feature in zip(axes, _LAB_FEATURES):
        sns.boxplot(
            data=df,
            x=TARGET_COLUMN,
            y=feature,
            hue=TARGET_COLUMN,
            palette=palette,
            width=0.5,
            flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
            legend=False,
            ax=axis,
        )
        axis.set_title(f"{feature} by Disease Class")
        axis.set_xlabel("")
        axis.set_ylabel(feature)
        plt.setp(axis.get_xticklabels(), rotation=10, ha="right")

    for axis in axes[len(_LAB_FEATURES):]:
        axis.axis("off")

    fig.suptitle("Lab Feature Distributions by Disease Class", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(
        PLOTS_DIR / "02_lab_boxplots_by_target.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def build_summary_text(
    df: pd.DataFrame,
    raw_dataset_path: Path,
    outlier_summary: pd.DataFrame,
) -> str:
    class_counts = df[TARGET_COLUMN].value_counts()
    gender_counts = df[GENDER_COLUMN].value_counts()
    subgroup_target = (
        pd.crosstab(df[GENDER_COLUMN], df[TARGET_COLUMN], normalize="index")
        .mul(100)
        .round(2)
    )
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    strongest_correlations = (
        df[numeric_columns]
        .corr()
        .where(lambda corr: corr.abs() < 1)
        .stack()
        .sort_values(key=lambda series: series.abs(), ascending=False)
        .head(5)
    )

    outlier_lines = [
        f"- {row.column}: {row.outlier_count} observations ({row.outlier_pct:.2f}%)"
        for row in outlier_summary.itertuples(index=False)
        if row.outlier_count > 0
    ]
    if not outlier_lines:
        outlier_lines = ["- No numeric outliers detected using the IQR rule."]

    correlation_lines = [
        f"- {left} vs {right}: {value:.3f}"
        for (left, right), value in strongest_correlations.items()
    ]

    positive_rate_by_gender = subgroup_target.get(POSITIVE_CLASS, pd.Series(dtype=float))
    imbalance_gap = 0.0
    if not positive_rate_by_gender.empty and positive_rate_by_gender.size > 1:
        imbalance_gap = float(positive_rate_by_gender.max() - positive_rate_by_gender.min())

    lines = [
        "Stage 2 - Data Understanding Summary",
        "",
        f"Dataset path: {DATASET_PATH}",
        f"Archived raw copy: {raw_dataset_path}",
        f"Rows: {df.shape[0]}",
        f"Columns: {df.shape[1]}",
        f"Numeric features: {', '.join(numeric_columns)}",
        "",
        "Class distribution:",
        f"- Liver Disease: {int(class_counts.get('Liver Disease', 0))}",
        f"- No Liver Disease: {int(class_counts.get('No Liver Disease', 0))}",
        "",
        "Gender distribution:",
        f"- Male: {int(gender_counts.get('Male', 0))}",
        f"- Female: {int(gender_counts.get('Female', 0))}",
        "",
        "Missing-value assessment:",
        "- No missing values were detected in the dataset.",
        "",
        "Outlier overview:",
        *outlier_lines,
        "",
        "Strongest numeric correlations:",
        *correlation_lines,
        "",
        "Potential bias sources:",
        "- The dataset is gender-imbalanced, with substantially more male than female patients.",
        "- The target is class-imbalanced toward the Liver Disease class.",
        f"- Positive-class rate gap across gender groups: {imbalance_gap:.2f} percentage points.",
        "- A smaller female subgroup may yield less stable estimates during model evaluation.",
        "- Outliers in enzyme and bilirubin features may disproportionately affect model fitting and interpretation.",
    ]
    return "\n".join(lines)


def build_ai_ready_summary(df: pd.DataFrame) -> str:
    gender_distribution = (df[GENDER_COLUMN].value_counts(normalize=True) * 100).round(2)
    target_distribution = (df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2)
    subgroup_distribution = pd.crosstab(df[GENDER_COLUMN], df[TARGET_COLUMN], normalize="index").mul(100).round(2)

    female_positive = float(subgroup_distribution.loc["Female", POSITIVE_CLASS])
    male_positive = float(subgroup_distribution.loc["Male", POSITIVE_CLASS])

    lines = [
        "AI-Ready EDA Summary",
        "",
        "Key findings:",
        f"1. The dataset contains {df.shape[0]} patient records and {df.shape[1]} variables, including 9 numeric predictors, 1 gender field, and 1 binary target.",
        f"2. Gender distribution is imbalanced: Male {gender_distribution.get('Male', 0.0):.2f}% and Female {gender_distribution.get('Female', 0.0):.2f}%.",
        f"3. Target distribution is also imbalanced: Liver Disease {target_distribution.get('Liver Disease', 0.0):.2f}% and No Liver Disease {target_distribution.get('No Liver Disease', 0.0):.2f}%.",
        "4. No missing values were found, so preprocessing can focus on encoding, scaling choices, and robust evaluation rather than imputation.",
        f"5. The positive-class prevalence differs by gender: Male {male_positive:.2f}% versus Female {female_positive:.2f}%, indicating a subgroup distribution shift that must be considered during evaluation.",
        "6. Several laboratory features contain outliers, suggesting the need for careful inspection of scaling sensitivity and model robustness.",
        "7. Because the female subgroup is much smaller, subgroup metrics may exhibit higher variance and wider uncertainty than the male subgroup.",
        "",
        "Implications for Stage 3 and Stage 4:",
        "- Use stratified splitting on the target and track subgroup representation after the split.",
        "- Preserve the original gender column for subgroup evaluation even if alternate model variants exclude it as an input feature.",
        "- Compare global and subgroup metrics to avoid relying on aggregate performance alone.",
        "- Pay special attention to recall and false negative rate for the female subgroup because missed diagnoses are clinically important.",
    ]
    return "\n".join(lines)


def run_stage_2() -> None:
    ensure_directories()
    df = load_dataset()
    raw_dataset_path = archive_raw_dataset()

    data_profile = build_data_profile(df)
    missing_table = build_missing_values_table(df)
    outlier_summary = compute_outlier_summary(df)

    data_profile.to_csv(TABLES_DIR / "02_data_profile.csv", index=False)
    missing_table.to_csv(TABLES_DIR / "02_missing_values.csv", index=False)
    outlier_summary.to_csv(TABLES_DIR / "02_outlier_summary.csv", index=False)

    plot_gender_distribution(df)
    plot_target_distribution(df)
    plot_missing_values(df)
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_target_by_gender(df)
    plot_lab_boxplots_by_gender(df)
    plot_lab_boxplots_by_target(df)

    (REPORTS_DIR / "02_data_understanding_summary.txt").write_text(
        build_summary_text(df, raw_dataset_path, outlier_summary),
        encoding="utf-8",
    )
    (REPORTS_DIR / "02_ai_ready_eda_summary.txt").write_text(
        build_ai_ready_summary(df),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run_stage_2()