from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.model_selection import cross_validate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import TABLES_DIR  # noqa: E402
from src.rq2_baseline import (  # noqa: E402
    RQ2_FEATURE_SET_A,
    RQ2_FEATURE_SET_B,
    SCORING,
    build_cv_splitter,
    build_delta_table,
    build_model_pipelines,
    build_split_data,
    evaluate_on_test_set,
    load_modeling_frame,
    make_patient_aware_split,
)
from src.rq2_visualization import plot_rq2_feature_comparison, read_delta_pr_auc, read_test_pr_auc  # noqa: E402


REMOVED_FEATURE = "discharge_disposition_group"

SENSITIVITY_FEATURE_SETS = {
    "A": [feature for feature in RQ2_FEATURE_SET_A if feature != REMOVED_FEATURE],
    "B": [feature for feature in RQ2_FEATURE_SET_B if feature != REMOVED_FEATURE],
}

CV_RESULTS_PATH = TABLES_DIR / "rq2_sensitivity_no_discharge_cv_results.csv"
TEST_RESULTS_PATH = TABLES_DIR / "rq2_sensitivity_no_discharge_test_results.csv"
DELTA_RESULTS_PATH = TABLES_DIR / "rq2_sensitivity_no_discharge_delta_results.csv"
SPLIT_SUMMARY_PATH = TABLES_DIR / "rq2_sensitivity_no_discharge_split_summary.csv"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "rq2_figure_02_sensitivity_no_discharge.png"


def run_sensitivity_no_discharge() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_feature_columns = max(SENSITIVITY_FEATURE_SETS.values(), key=len)
    _, y, groups = load_modeling_frame(base_feature_columns)
    split_indices = make_patient_aware_split(y, groups)
    cv_splitter, cv_name = build_cv_splitter()

    cv_rows: list[dict[str, float | str]] = []
    test_rows: list[dict[str, float | str]] = []

    for feature_set_name, feature_columns in SENSITIVITY_FEATURE_SETS.items():
        X, _, _ = load_modeling_frame(feature_columns)
        split_data = build_split_data(X, y, groups, split_indices)
        model_pipelines = build_model_pipelines(feature_columns)

        for model_name, pipeline in model_pipelines.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                scores = cross_validate(
                    pipeline,
                    split_data.X_train,
                    split_data.y_train,
                    groups=split_data.groups_train,
                    cv=cv_splitter,
                    scoring=SCORING,
                    n_jobs=1,
                    error_score="raise",
                )

            cv_rows.append(
                {
                    "model": model_name,
                    "feature_set": feature_set_name,
                    "pr_auc_mean": scores["test_pr_auc"].mean(),
                    "pr_auc_std": scores["test_pr_auc"].std(),
                    "recall_mean": scores["test_recall_30d"].mean(),
                    "f1_mean": scores["test_f1_30d"].mean(),
                    "roc_auc_mean": scores["test_roc_auc"].mean(),
                }
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                test_row = evaluate_on_test_set(model_name, pipeline, split_data)
                test_row["feature_set"] = feature_set_name
                test_rows.append(test_row)

    split_summary = pd.DataFrame(
        [
            {
                "removed_feature": REMOVED_FEATURE,
                "split_method": split_indices.split_method,
                "cv_method": cv_name,
                "n_train_encounters": len(split_indices.train_idx),
                "n_test_encounters": len(split_indices.test_idx),
                "n_train_patients": groups.iloc[split_indices.train_idx].nunique(),
                "n_test_patients": groups.iloc[split_indices.test_idx].nunique(),
                "train_positive_rate": y.iloc[split_indices.train_idx].mean(),
                "test_positive_rate": y.iloc[split_indices.test_idx].mean(),
                "patient_overlap": int(
                    len(
                        set(groups.iloc[split_indices.train_idx]).intersection(
                            set(groups.iloc[split_indices.test_idx])
                        )
                    )
                ),
            }
        ]
    )

    cv_results = (
        pd.DataFrame(cv_rows)
        .sort_values(["feature_set", "pr_auc_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )
    test_results = (
        pd.DataFrame(test_rows)[
            ["model", "feature_set", "pr_auc", "recall_30d", "precision_30d", "f1_30d", "roc_auc"]
        ]
        .sort_values(["feature_set", "pr_auc"], ascending=[True, False])
        .reset_index(drop=True)
    )
    delta_table = build_delta_table(test_results)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(CV_RESULTS_PATH, index=False)
    test_results.to_csv(TEST_RESULTS_PATH, index=False)
    delta_table.to_csv(DELTA_RESULTS_PATH, index=False)
    split_summary.to_csv(SPLIT_SUMMARY_PATH, index=False)

    return cv_results, test_results, delta_table, split_summary


def main() -> None:
    cv_results, test_results, delta_table, split_summary = run_sensitivity_no_discharge()
    pr_auc = read_test_pr_auc(TEST_RESULTS_PATH)
    deltas = read_delta_pr_auc(DELTA_RESULTS_PATH)
    plot_rq2_feature_comparison(
        pr_auc,
        deltas,
        FIGURE_PATH,
        title="RQ2 sensitivity: no discharge disposition feature",
    )

    print(f"Removed feature: {REMOVED_FEATURE}")
    print(f"Wrote {CV_RESULTS_PATH}")
    print(f"Wrote {TEST_RESULTS_PATH}")
    print(f"Wrote {DELTA_RESULTS_PATH}")
    print(f"Wrote {SPLIT_SUMMARY_PATH}")
    print(f"Wrote {FIGURE_PATH}")
    print()
    print(split_summary.to_string(index=False))
    print()
    print(test_results.to_string(index=False))
    print()
    print(delta_table.to_string(index=False))


if __name__ == "__main__":
    main()
