from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_PATH = RAW_DIR / "diabetic_data.csv"
MAPPING_PATH = RAW_DIR / "IDS_mapping.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

EXPIRED_DISCHARGE_IDS = {11, 19, 20, 21}
HOSPICE_DISCHARGE_IDS = {13, 14}

DROP_COLUMNS = ["weight", "payer_code"]
MODEL_EXCLUDED_IDENTIFIER_COLUMNS = ["encounter_id", "patient_nbr"]

DRUG_COLUMNS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]


def parse_id_mapping(path: Path = MAPPING_PATH) -> dict[str, dict[int, str]]:
    """Parse the UCI IDS_mapping.csv file, which stores three mappings in one file."""
    mappings: dict[str, dict[int, str]] = {}
    current_name: str | None = None

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or all(not cell.strip() for cell in row):
                current_name = None
                continue

            first = row[0].strip()
            if first in {
                "admission_type_id",
                "discharge_disposition_id",
                "admission_source_id",
            }:
                current_name = first
                mappings[current_name] = {}
                continue

            if current_name is None or len(row) < 2 or not first:
                continue

            try:
                key = int(first)
            except ValueError:
                continue
            mappings[current_name][key] = row[1].strip()

    return mappings


def group_admission_type(value: float | int | str) -> str:
    if pd.isna(value):
        return "Missing/Unknown"
    value = int(value)
    if value == 1:
        return "Emergency"
    if value == 2:
        return "Urgent"
    if value == 3:
        return "Elective"
    if value == 7:
        return "Trauma"
    return "Other/Unknown"


def group_admission_source(value: float | int | str) -> str:
    if pd.isna(value):
        return "Missing/Unknown"
    value = int(value)
    if value == 7:
        return "Emergency Room"
    if value in {1, 2, 3}:
        return "Referral"
    if value in {4, 5, 6, 10, 18, 19, 22, 25, 26}:
        return "Transfer"
    return "Other/Unknown"


def group_discharge_disposition(value: float | int | str) -> str:
    if pd.isna(value):
        return "Missing/Unknown"
    value = int(value)
    if value in EXPIRED_DISCHARGE_IDS:
        return "Expired"
    if value in HOSPICE_DISCHARGE_IDS:
        return "Hospice"
    if value == 1:
        return "Home"
    if value in {6, 8}:
        return "Home health/IV"
    if value in {2, 3, 4, 5, 15, 22, 23, 24, 27, 28, 29, 30}:
        return "Transferred facility"
    return "Other/Unknown"


def group_icd9(code: object) -> str:
    """Coarse ICD-9 grouping for compact RQ1/RQ2 features."""
    if pd.isna(code):
        return "Unknown"

    text = str(code).strip()
    if not text:
        return "Unknown"
    if text.startswith("V") or text.startswith("E"):
        return "Other"

    try:
        value = float(text)
    except ValueError:
        return "Unknown"

    if 250 <= value < 251:
        return "Diabetes"
    if 390 <= value <= 459 or value == 785:
        return "Circulatory"
    if 460 <= value <= 519 or value == 786:
        return "Respiratory"
    if 520 <= value <= 579 or value == 787:
        return "Digestive"
    if 800 <= value <= 999:
        return "Injury"
    if 710 <= value <= 739:
        return "Musculoskeletal"
    if 580 <= value <= 629 or value == 788:
        return "Genitourinary"
    if 140 <= value <= 239:
        return "Neoplasms"
    return "Other"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mappings = parse_id_mapping()

    for column in ["admission_type_id", "discharge_disposition_id", "admission_source_id"]:
        df[f"{column}_description"] = df[column].map(mappings[column]).fillna("Missing/Unknown")

    df["admission_type_group"] = df["admission_type_id"].apply(group_admission_type)
    df["admission_source_group"] = df["admission_source_id"].apply(group_admission_source)
    df["discharge_disposition_group"] = df["discharge_disposition_id"].apply(
        group_discharge_disposition
    )

    df["is_expired_discharge"] = df["discharge_disposition_id"].isin(EXPIRED_DISCHARGE_IDS)
    df["is_hospice_discharge"] = df["discharge_disposition_id"].isin(HOSPICE_DISCHARGE_IDS)

    for column in ["diag_1", "diag_2", "diag_3"]:
        df[f"{column}_group"] = df[column].apply(group_icd9)

    diag_groups = df[["diag_1_group", "diag_2_group", "diag_3_group"]]
    df["has_diabetes_diag"] = diag_groups.eq("Diabetes").any(axis=1)

    utilisation_columns = ["number_outpatient", "number_emergency", "number_inpatient"]
    df["prior_utilisation"] = df[utilisation_columns].sum(axis=1)
    df["has_prior_outpatient"] = df["number_outpatient"] > 0
    df["has_prior_emergency"] = df["number_emergency"] > 0
    df["has_prior_inpatient"] = df["number_inpatient"] > 0

    available_drug_columns = [column for column in DRUG_COLUMNS if column in df.columns]
    df["active_diabetes_med_count"] = df[available_drug_columns].ne("No").sum(axis=1)
    df["changed_diabetes_med_count"] = df[available_drug_columns].isin(["Up", "Down"]).sum(axis=1)
    df["early_readmission"] = (df["readmitted"] == "<30").astype(int)

    return df


def build_cleaned_dataset(
    raw_path: Path = RAW_DATA_PATH,
    exclude_expired: bool = True,
    exclude_hospice: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(raw_path, na_values="?", low_memory=False)
    cleaned = add_engineered_features(raw)

    if exclude_expired:
        cleaned = cleaned.loc[~cleaned["is_expired_discharge"]].copy()
    if exclude_hospice:
        cleaned = cleaned.loc[~cleaned["is_hospice_discharge"]].copy()

    cleaned = cleaned.drop(columns=[column for column in DROP_COLUMNS if column in cleaned.columns])

    cohort_rows = [
        {
            "step": "Raw encounters",
            "n_encounters": len(raw),
            "n_unique_patients": raw["patient_nbr"].nunique(),
        },
        {
            "step": "After excluding expired discharges",
            "n_encounters": len(cleaned),
            "n_unique_patients": cleaned["patient_nbr"].nunique(),
        },
    ]

    if exclude_hospice:
        cohort_rows.append(
            {
                "step": "After excluding hospice discharges",
                "n_encounters": len(cleaned),
                "n_unique_patients": cleaned["patient_nbr"].nunique(),
            }
        )

    return cleaned, pd.DataFrame(cohort_rows)


def write_outputs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    cleaned, cohort_summary = build_cleaned_dataset()
    raw = pd.read_csv(RAW_DATA_PATH, na_values="?", low_memory=False)

    cleaned_path = PROCESSED_DIR / "cleaned_diabetic_encounters.csv"
    cohort_path = TABLE_DIR / "cohort_summary.csv"
    missing_path = TABLE_DIR / "raw_missing_summary.csv"
    discharge_path = TABLE_DIR / "expired_hospice_discharge_summary.csv"
    identifier_path = TABLE_DIR / "model_feature_exclusion_notes.csv"

    cleaned.to_csv(cleaned_path, index=False)
    cohort_summary.to_csv(cohort_path, index=False)

    missing_summary = (
        raw.isna()
        .sum()
        .rename("missing_count")
        .to_frame()
        .assign(missing_percent=lambda x: x["missing_count"] / len(raw) * 100)
        .query("missing_count > 0")
        .sort_values("missing_percent", ascending=False)
    )
    missing_summary.to_csv(missing_path)

    discharge_interest = raw.loc[
        raw["discharge_disposition_id"].isin(EXPIRED_DISCHARGE_IDS | HOSPICE_DISCHARGE_IDS)
    ]
    discharge_summary = (
        pd.crosstab(discharge_interest["discharge_disposition_id"], discharge_interest["readmitted"])
        .rename_axis(index="discharge_disposition_id")
        .reset_index()
    )
    discharge_summary["discharge_group"] = discharge_summary["discharge_disposition_id"].apply(
        group_discharge_disposition
    )
    discharge_summary.to_csv(discharge_path, index=False)

    identifier_notes = pd.DataFrame(
        [
            {
                "column": "encounter_id",
                "cleaned_csv_status": "retained for traceability",
                "modelling_status": "exclude from model features",
                "reason": "unique encounter identifier with no predictive clinical meaning",
            },
            {
                "column": "patient_nbr",
                "cleaned_csv_status": "retained for patient-aware splitting",
                "modelling_status": "use for split only; exclude from model features",
                "reason": "patient identifier can cause leakage if used as a feature",
            },
        ]
    )
    identifier_notes.to_csv(identifier_path, index=False)

    print(f"Wrote {cleaned_path.relative_to(PROJECT_ROOT)} with shape {cleaned.shape}")
    print(f"Wrote {cohort_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {missing_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {discharge_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {identifier_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    write_outputs()
