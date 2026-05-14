import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_summary(name, y_true, y_pred, y_score=None):
    """Create one result row focused on the readmission-within-30-days class."""
    row = {
        "model": name,
        "precision_30d": precision_score(y_true, y_pred, zero_division=0),
        "recall_30d": recall_score(y_true, y_pred, zero_division=0),
        "f1_30d": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_score is not None:
        row["pr_auc"] = average_precision_score(y_true, y_score)
        row["roc_auc"] = roc_auc_score(y_true, y_score)

    return pd.DataFrame([row])
