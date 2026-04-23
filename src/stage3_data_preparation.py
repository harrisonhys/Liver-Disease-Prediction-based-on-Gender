from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.config import (
    DATASET_PATH,
    GENDER_COLUMN,
    POSITIVE_CLASS,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    TARGET_COLUMN,
    ensure_directories,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Features to apply log1p before scaling
LOG_FEATURES = ["TB", "DB", "Alkphos", "Sgpt", "Sgot"]

# All numeric input features (original + engineered)
NUMERIC_FEATURES = ["Age", "TB", "DB", "Alkphos", "Sgpt", "Sgot", "TP", "ALB", "A/G Ratio"]
ENGINEERED_FEATURES = ["fractional_bilirubin"]
ALL_FEATURES: list[str] = NUMERIC_FEATURES + ENGINEERED_FEATURES + ["Gender_encoded"]


def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET_PATH)


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map target to binary int: Liver Disease → 1, else → 0."""
    df = df.copy()
    df["target"] = (df[TARGET_COLUMN] == POSITIVE_CLASS).astype(int)
    return df


def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Binary encode Gender: Male → 1, Female → 0. Keep original column."""
    df = df.copy()
    df["Gender_encoded"] = (df[GENDER_COLUMN] == "Male").astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add fractional bilirubin (DB/TB ratio) as a clinically-grounded feature."""
    df = df.copy()
    df["fractional_bilirubin"] = df["DB"] / (df["TB"] + 1e-6)
    return df


def apply_log_transform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Apply log1p to right-skewed enzyme/bilirubin columns."""
    df = df.copy()
    for col in columns:
        df[col] = np.log1p(df[col])
    return df


def stratified_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split using a 4-class stratification key (Gender × Target) so that all
    four subgroups are proportionally represented in both train and test sets.
    This prevents the Female No-Liver-Disease minority from being wiped out
    of the test set under a simple target-only stratification.
    """
    strat_key = df[GENDER_COLUMN] + "_" + df["target"].astype(str)
    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=strat_key,
        random_state=RANDOM_STATE,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def fit_and_apply_scaler(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, RobustScaler]:
    """Fit RobustScaler on train only, then transform both splits."""
    scaler = RobustScaler()
    train = train.copy()
    test = test.copy()
    train[features] = scaler.fit_transform(train[features])
    test[features] = scaler.transform(test[features])
    return train, test, scaler


def build_subgroup_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute describe() per (Gender × Target) subgroup and stack into one table."""
    rows = []
    lab_cols = NUMERIC_FEATURES + ENGINEERED_FEATURES
    for gender in ["Male", "Female"]:
        for label, label_str in [(1, POSITIVE_CLASS), (0, "No Liver Disease")]:
            sub = df[(df[GENDER_COLUMN] == gender) & (df["target"] == label)][lab_cols]
            desc = sub.describe(percentiles=[0.25, 0.5, 0.75]).T.round(3)
            desc.insert(0, "subgroup", f"{gender} — {label_str}")
            desc.index.name = "feature"
            rows.append(desc.reset_index())
    return pd.concat(rows, ignore_index=True)


def build_split_balance_table(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for split_name, split_df in [("train", train), ("test", test)]:
        counts = (
            split_df.groupby([GENDER_COLUMN, "target"])
            .size()
            .reset_index(name="count")
        )
        counts["split"] = split_name
        rows.append(counts)
    balance = pd.concat(rows, ignore_index=True)
    balance["target_label"] = balance["target"].map({1: POSITIVE_CLASS, 0: "No Liver Disease"})
    totals = balance.groupby("split")["count"].transform("sum")
    balance["pct_of_split"] = (balance["count"] / totals * 100).round(2)
    return balance[["split", GENDER_COLUMN, "target_label", "count", "pct_of_split"]]


def build_preprocessing_steps_log(
    n_train: int,
    n_test: int,
    feature_list: list[str],
    log_features: list[str],
    engineered: list[str],
) -> str:
    lines = [
        "Stage 3 - Data Preparation Steps",
        "",
        "1. Target encoding",
        f"   - '{POSITIVE_CLASS}' → 1,  'No Liver Disease' → 0",
        "",
        "2. Gender encoding",
        "   - Male → 1,  Female → 0  (binary label encoding)",
        "   - Original Gender column preserved for subgroup tracking",
        "",
        "3. Feature engineering",
        f"   - Added: {', '.join(engineered)}",
        "   - fractional_bilirubin = DB / (TB + 1e-6)  (clinically grounded DB/TB ratio)",
        "",
        "4. Log transformation (log1p)",
        f"   - Applied to: {', '.join(log_features)}",
        "   - Rationale: right-skewed, heavy-tailed distributions; reduces outlier influence",
        "",
        "5. Train-test split",
        "   - Strategy: stratified by Gender × Target (4-class key)",
        f"   - Test size: {TEST_SIZE * 100:.0f}%  |  Random state: {RANDOM_STATE}",
        f"   - Train: {n_train} rows  |  Test: {n_test} rows",
        "",
        "6. Feature scaling",
        "   - Scaler: RobustScaler (median + IQR)",
        "   - Rationale: resistant to extreme enzyme/bilirubin outliers",
        "   - Fit on training set only; applied to test set without data leakage",
        "",
        f"7. Final feature list ({len(feature_list)} features):",
        *[f"   - {f}" for f in feature_list],
    ]
    return "\n".join(lines)


def build_ai_ready_summary(
    df: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> str:
    female_neg_test = int(
        ((test[GENDER_COLUMN] == "Female") & (test["target"] == 0)).sum()
    )
    lines = [
        "AI-Ready Preprocessing Summary",
        "",
        "Actions taken:",
        "1. Binary target encoding: Liver Disease=1, No Liver Disease=0.",
        "2. Binary gender encoding: Male=1, Female=0. Original Gender column kept for subgroup evaluation.",
        "3. Engineered feature 'fractional_bilirubin' (DB/TB ratio) added as a clinically grounded indicator.",
        f"4. Log1p applied to {', '.join(LOG_FEATURES)} to compress right-skewed distributions.",
        f"5. 4-class stratified split (Gender × Target): {len(train)} train / {len(test)} test.",
        f"   - Female No-Liver-Disease samples in test set: {female_neg_test} (protected by 4-class stratification).",
        "6. RobustScaler fitted on training set; applied to test set to prevent data leakage.",
        "",
        "Artifact locations:",
        "  - data/processed/train.csv          — scaled training data with all features + target + Gender",
        "  - data/processed/test.csv           — scaled test data",
        "  - data/processed/train_raw.csv      — unscaled training data (for SHAP/interpretation use)",
        "  - data/processed/test_raw.csv       — unscaled test data",
        "  - results/tables/03_selected_features.csv",
        "  - results/tables/03_split_balance.csv",
        "  - results/tables/03_subgroup_descriptive_stats.csv",
        "",
        "Ready for Stage 4 modeling.",
        "Key note: always use the Gender column from train/test CSVs for all subgroup splits in Stage 4.",
    ]
    return "\n".join(lines)


def run_stage_3() -> None:
    ensure_directories()

    df = load_dataset()
    df = encode_target(df)
    df = encode_gender(df)
    df = engineer_features(df)
    df = apply_log_transform(df, LOG_FEATURES)

    train_raw, test_raw = stratified_split(df)

    # Scale numeric + engineered features (not the binary Gender_encoded or target)
    scale_cols = NUMERIC_FEATURES + ENGINEERED_FEATURES
    train_scaled, test_scaled, scaler = fit_and_apply_scaler(train_raw, test_raw, scale_cols)

    # Save processed datasets
    train_scaled.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    test_scaled.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    train_raw.to_csv(PROCESSED_DATA_DIR / "train_raw.csv", index=False)
    test_raw.to_csv(PROCESSED_DATA_DIR / "test_raw.csv", index=False)

    # Save feature list
    feature_df = pd.DataFrame({"feature": ALL_FEATURES})
    feature_df["type"] = feature_df["feature"].apply(
        lambda f: "engineered"
        if f in ENGINEERED_FEATURES
        else ("categorical_binary" if f == "Gender_encoded" else "numeric")
    )
    feature_df.to_csv(TABLES_DIR / "03_selected_features.csv", index=False)

    # Save split balance table
    split_balance = build_split_balance_table(train_scaled, test_scaled)
    split_balance.to_csv(TABLES_DIR / "03_split_balance.csv", index=False)

    # Save subgroup descriptive stats (computed on raw/unscaled values for interpretability)
    subgroup_stats = build_subgroup_descriptive_stats(df)
    subgroup_stats.to_csv(TABLES_DIR / "03_subgroup_descriptive_stats.csv", index=False)

    # Save scaler feature names and center/scale for reproducibility
    scaler_meta = {
        "features": scale_cols,
        "center_": scaler.center_.tolist(),
        "scale_": scaler.scale_.tolist(),
    }
    (PROCESSED_DATA_DIR / "scaler_meta.json").write_text(
        json.dumps(scaler_meta, indent=2), encoding="utf-8"
    )

    # Write reports
    (REPORTS_DIR / "03_data_preparation.txt").write_text(
        build_preprocessing_steps_log(
            n_train=len(train_scaled),
            n_test=len(test_scaled),
            feature_list=ALL_FEATURES,
            log_features=LOG_FEATURES,
            engineered=ENGINEERED_FEATURES,
        ),
        encoding="utf-8",
    )
    (REPORTS_DIR / "03_ai_ready_preprocessing_summary.txt").write_text(
        build_ai_ready_summary(df, train_scaled, test_scaled),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run_stage_3()
