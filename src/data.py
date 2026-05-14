import pandas as pd

from .paths import DIABETIC_DATA_PATH, IDS_MAPPING_PATH


def load_diabetic_data(path=DIABETIC_DATA_PATH):
    """Load the main diabetes readmission dataset."""
    return pd.read_csv(path)


def load_ids_mapping(path=IDS_MAPPING_PATH):
    """Load the ID mapping file for admission, discharge, and source IDs."""
    return pd.read_csv(path)


def make_binary_readmission_target(df):
    """Return a copy with target_30d = 1 for readmission within 30 days."""
    out = df.copy()
    out["target_30d"] = (out["readmitted"] == "<30").astype(int)
    return out
