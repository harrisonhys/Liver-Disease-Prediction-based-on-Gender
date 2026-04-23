from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "datasets" / "liver_patient_dataset.csv"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"

TARGET_COLUMN = "Selector"
GENDER_COLUMN = "Gender"
POSITIVE_CLASS = "Liver Disease"


def ensure_directories() -> None:
    for path in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
        TABLES_DIR,
        MODELS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)