from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import TABLES_DIR  # noqa: E402
from src.rq2_baseline import (  # noqa: E402
    DATA_PATH,
    RQ2_FEATURE_SET_B,
    build_model_pipelines,
    build_split_data,
    load_modeling_frame,
    make_patient_aware_split,
)


SELECTED_MODEL_NAME = "Random Forest"
SELECTED_FEATURE_SET = "B"
DECISION_THRESHOLD = 0.5

FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
OVERALL_METRICS_PATH = TABLES_DIR / "rq3_selected_model_overall_metrics.csv"
SUBGROUP_METRICS_PATH = TABLES_DIR / "rq3_subgroup_metrics.csv"
FALSE_NEGATIVE_PROFILE_PATH = TABLES_DIR / "rq3_false_negative_profile.csv"
SUBGROUP_SPREAD_PATH = TABLES_DIR / "rq3_subgroup_metric_spread.csv"
FIGURE_PATH = FIGURE_DIR / "rq3_figure_01_subgroup_false_negative_rates.png"


@dataclass(frozen=True)
class SubgroupDefinition:
    column: str
    label: str
    order: list[str]


SUBGROUP_DEFINITIONS = [
    SubgroupDefinition(
        column="age_group_rq3",
        label="Age group",
        order=["<50", "50-60", "60-70", "70-80", "80+"],
    ),
    SubgroupDefinition(
        column="race_group_rq3",
        label="Race group",
        order=["Caucasian", "AfricanAmerican", "Other/Unknown"],
    ),
    SubgroupDefinition(
        column="admission_source_group",
        label="Admission source",
        order=["Emergency Room", "Referral", "Transfer", "Other/Unknown"],
    ),
]


def collapse_age(value: object) -> str:
    text = str(value).strip()
    if text in {"[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)"}:
        return "<50"
    if text == "[50-60)":
        return "50-60"
    if text == "[60-70)":
        return "60-70"
    if text == "[70-80)":
        return "70-80"
    if text in {"[80-90)", "[90-100)"}:
        return "80+"
    return "Other/Unknown"


def collapse_race(value: object) -> str:
    if pd.isna(value):
        return "Other/Unknown"
    text = str(value).strip()
    if text in {"Caucasian", "AfricanAmerican"}:
        return text
    return "Other/Unknown"


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else np.nan


def binary_metrics(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true_array = y_true.to_numpy()
    tp = int(((y_true_array == 1) & (y_pred == 1)).sum())
    fn = int(((y_true_array == 1) & (y_pred == 0)).sum())
    fp = int(((y_true_array == 0) & (y_pred == 1)).sum())
    tn = int(((y_true_array == 0) & (y_pred == 0)).sum())
    n_positive = int(y_true_array.sum())
    n_negative = int(len(y_true_array) - n_positive)

    row: dict[str, float] = {
        "n": int(len(y_true_array)),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_rate": safe_divide(n_positive, len(y_true_array)),
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "recall_30d": recall_score(y_true_array, y_pred, zero_division=0),
        "false_negative_rate": safe_divide(fn, tp + fn),
        "precision_30d": precision_score(y_true_array, y_pred, zero_division=0),
        "f1_30d": f1_score(y_true_array, y_pred, zero_division=0),
        "specificity": safe_divide(tn, tn + fp),
        "false_positive_rate": safe_divide(fp, tn + fp),
        "predicted_positive_rate": float(y_pred.mean()),
        "mean_predicted_risk": float(np.mean(y_score)),
        "median_predicted_risk": float(np.median(y_score)),
        "accuracy": accuracy_score(y_true_array, y_pred),
    }

    if n_positive > 0 and n_negative > 0:
        row["pr_auc"] = average_precision_score(y_true_array, y_score)
        row["roc_auc"] = roc_auc_score(y_true_array, y_score)
    else:
        row["pr_auc"] = np.nan
        row["roc_auc"] = np.nan

    return row


def prepare_analysis_frame(
    split_data,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    split_indices,
) -> pd.DataFrame:
    full_df = pd.read_csv(DATA_PATH, low_memory=False)
    test_meta = full_df.iloc[split_indices.test_idx].reset_index(drop=True).copy()
    analysis_df = test_meta[
        ["age", "race", "admission_source_group", "early_readmission", "patient_nbr"]
    ].copy()
    analysis_df["y_true"] = split_data.y_test.to_numpy()
    analysis_df["y_score"] = y_score
    analysis_df["y_pred"] = y_pred
    analysis_df["age_group_rq3"] = analysis_df["age"].apply(collapse_age)
    analysis_df["race_group_rq3"] = analysis_df["race"].apply(collapse_race)
    return analysis_df


def build_subgroup_metrics(analysis_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for definition in SUBGROUP_DEFINITIONS:
        for order_index, subgroup in enumerate(definition.order):
            subset = analysis_df.loc[analysis_df[definition.column] == subgroup]
            if subset.empty:
                continue
            metrics = binary_metrics(
                subset["y_true"].astype(int),
                subset["y_pred"].to_numpy(dtype=int),
                subset["y_score"].to_numpy(dtype=float),
            )
            rows.append(
                {
                    "subgroup_variable": definition.label,
                    "subgroup": subgroup,
                    "subgroup_order": order_index,
                    "low_positive_support": bool(metrics["n_positive"] < 50),
                    **metrics,
                }
            )

    return pd.DataFrame(rows).sort_values(["subgroup_variable", "subgroup_order"])


def build_false_negative_profile(analysis_df: pd.DataFrame) -> pd.DataFrame:
    positive_df = analysis_df.loc[analysis_df["y_true"] == 1].copy()
    total_positives = len(positive_df)
    total_false_negatives = int((positive_df["y_pred"] == 0).sum())

    rows: list[dict[str, object]] = []
    for definition in SUBGROUP_DEFINITIONS:
        for order_index, subgroup in enumerate(definition.order):
            subset = positive_df.loc[positive_df[definition.column] == subgroup]
            if subset.empty:
                continue
            false_negatives = int((subset["y_pred"] == 0).sum())
            share_of_all_positives = safe_divide(len(subset), total_positives)
            share_of_all_false_negatives = safe_divide(false_negatives, total_false_negatives)
            rows.append(
                {
                    "subgroup_variable": definition.label,
                    "subgroup": subgroup,
                    "subgroup_order": order_index,
                    "n_actual_positive": int(len(subset)),
                    "false_negatives": false_negatives,
                    "false_negative_rate": safe_divide(false_negatives, len(subset)),
                    "share_of_all_positives": share_of_all_positives,
                    "share_of_all_false_negatives": share_of_all_false_negatives,
                    "false_negative_representation_ratio": safe_divide(
                        share_of_all_false_negatives,
                        share_of_all_positives,
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values(["subgroup_variable", "subgroup_order"])


def build_subgroup_spread(subgroup_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subgroup_variable, group in subgroup_metrics.groupby("subgroup_variable", sort=False):
        rows.append(
            {
                "subgroup_variable": subgroup_variable,
                "min_recall_30d": group["recall_30d"].min(),
                "max_recall_30d": group["recall_30d"].max(),
                "recall_range": group["recall_30d"].max() - group["recall_30d"].min(),
                "min_false_negative_rate": group["false_negative_rate"].min(),
                "max_false_negative_rate": group["false_negative_rate"].max(),
                "false_negative_rate_range": (
                    group["false_negative_rate"].max() - group["false_negative_rate"].min()
                ),
                "min_pr_auc": group["pr_auc"].min(),
                "max_pr_auc": group["pr_auc"].max(),
                "pr_auc_range": group["pr_auc"].max() - group["pr_auc"].min(),
            }
        )
    return pd.DataFrame(rows)


def plot_false_negative_rates(subgroup_metrics: pd.DataFrame, overall_fnr: float) -> None:
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, len(SUBGROUP_DEFINITIONS), figsize=(14, 4.6), sharex=True)
    fig.patch.set_facecolor("white")

    for ax, definition in zip(axes, SUBGROUP_DEFINITIONS, strict=True):
        subset = subgroup_metrics.loc[subgroup_metrics["subgroup_variable"] == definition.label].copy()
        subset = subset.sort_values("subgroup_order", ascending=False)
        colors = ["#4C78A8" if not value else "#B279A2" for value in subset["low_positive_support"]]
        ax.barh(subset["subgroup"], subset["false_negative_rate"], color=colors)
        ax.axvline(overall_fnr, color="#333333", linewidth=1, linestyle="--")
        ax.set_title(definition.label)
        ax.set_xlabel("False-negative rate")
        ax.set_xlim(0, 0.65)
        ax.grid(axis="x", alpha=0.25)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        for index, row in enumerate(subset.itertuples(index=False)):
            ax.text(
                row.false_negative_rate + 0.01,
                index,
                f"{row.false_negative_rate:.2f} (n+={int(row.n_positive)})",
                va="center",
                ha="left",
                fontsize=8.5,
            )

    fig.suptitle(
        "RQ3: False-negative rates by subgroup\n"
        "Random Forest with Feature Set B, fixed 0.5 decision threshold",
        fontsize=13,
        y=1.04,
    )
    fig.tight_layout(w_pad=2.0)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_rq3_subgroup_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X, y, groups = load_modeling_frame(RQ2_FEATURE_SET_B)
    split_indices = make_patient_aware_split(y, groups)
    split_data = build_split_data(X, y, groups, split_indices)

    pipeline = build_model_pipelines(RQ2_FEATURE_SET_B)[SELECTED_MODEL_NAME]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pipeline.fit(split_data.X_train, split_data.y_train)

    y_score = pipeline.predict_proba(split_data.X_test)[:, 1]
    y_pred = (y_score >= DECISION_THRESHOLD).astype(int)

    overall_metrics = pd.DataFrame(
        [
            {
                "model": SELECTED_MODEL_NAME,
                "feature_set": SELECTED_FEATURE_SET,
                "decision_threshold": DECISION_THRESHOLD,
                "split_method": split_indices.split_method,
                "n_train_encounters": len(split_indices.train_idx),
                "n_test_encounters": len(split_indices.test_idx),
                "n_train_patients": groups.iloc[split_indices.train_idx].nunique(),
                "n_test_patients": groups.iloc[split_indices.test_idx].nunique(),
                "patient_overlap": int(
                    len(
                        set(groups.iloc[split_indices.train_idx]).intersection(
                            set(groups.iloc[split_indices.test_idx])
                        )
                    )
                ),
                **binary_metrics(split_data.y_test.astype(int), y_pred, y_score),
            }
        ]
    )

    analysis_df = prepare_analysis_frame(split_data, y_score, y_pred, split_indices)
    subgroup_metrics = build_subgroup_metrics(analysis_df)
    false_negative_profile = build_false_negative_profile(analysis_df)
    subgroup_spread = build_subgroup_spread(subgroup_metrics)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    overall_metrics.to_csv(OVERALL_METRICS_PATH, index=False)
    subgroup_metrics.to_csv(SUBGROUP_METRICS_PATH, index=False)
    false_negative_profile.to_csv(FALSE_NEGATIVE_PROFILE_PATH, index=False)
    subgroup_spread.to_csv(SUBGROUP_SPREAD_PATH, index=False)
    plot_false_negative_rates(
        subgroup_metrics,
        float(overall_metrics.loc[0, "false_negative_rate"]),
    )

    return overall_metrics, subgroup_metrics, false_negative_profile, subgroup_spread


def main() -> None:
    overall_metrics, subgroup_metrics, false_negative_profile, subgroup_spread = (
        run_rq3_subgroup_analysis()
    )

    print(f"Wrote {OVERALL_METRICS_PATH}")
    print(f"Wrote {SUBGROUP_METRICS_PATH}")
    print(f"Wrote {FALSE_NEGATIVE_PROFILE_PATH}")
    print(f"Wrote {SUBGROUP_SPREAD_PATH}")
    print(f"Wrote {FIGURE_PATH}")
    print()
    print(overall_metrics.to_string(index=False))
    print()
    print(subgroup_metrics.to_string(index=False))
    print()
    print(false_negative_profile.to_string(index=False))
    print()
    print(subgroup_spread.to_string(index=False))


if __name__ == "__main__":
    main()
