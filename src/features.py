import numpy as np


DIABETES_MEDICATION_COLUMNS = [
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


def add_engineered_features(df):
    """Add clinically motivated features used across experiments."""
    out = df.copy()
    out = out.replace("?", np.nan)

    out["prior_utilisation"] = (
        out["number_outpatient"] + out["number_emergency"] + out["number_inpatient"]
    )
    out["medication_change"] = (out["change"] == "Ch").astype(int)
    out["diabetes_medication_used"] = (out["diabetesMed"] == "Yes").astype(int)

    med_cols = [col for col in DIABETES_MEDICATION_COLUMNS if col in out.columns]
    out["treatment_complexity"] = out[med_cols].ne("No").sum(axis=1)

    return out
