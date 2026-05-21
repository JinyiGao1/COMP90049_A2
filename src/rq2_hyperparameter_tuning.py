from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".cache"))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.paths import FIGURES_DIR, TABLES_DIR
from src.rq2_baseline import (
    RANDOM_STATE,
    RQ2_FEATURE_SET_B,
    SCORING,
    build_preprocessor,
    load_modeling_frame,
    make_patient_aware_split,
)


TUNING_TABLE_PATH = TABLES_DIR / "rq2_hyperparameter_tuning_results.csv"
TUNING_SUMMARY_PATH = TABLES_DIR / "rq2_hyperparameter_tuning_summary.csv"
TUNING_FIGURE_PATH = FIGURES_DIR / "rq2_figure_01_hyperparameter_tuning.png"


# Tune one main hyperparameter per model
TUNING_GRID = {
    "Logistic Regression": {
        "parameter": "C",
        "values": [0.01, 0.1, 1.0, 10.0],
    },
    "Linear SVM": {
        "parameter": "C",
        "values": [0.001, 0.01, 0.1, 1.0],
    },
    "Multinomial Naive Bayes": {
        "parameter": "alpha",
        "values": [0.1, 1.0, 5.0, 10.0],
    },
    "Decision Tree": {
        "parameter": "max_depth",
        "values": [4, 8, 12, 20],
    },
    "Random Forest": {
        "parameter": "max_depth",
        "values": [6, 12, 20],
    },
    "MLP": {
        "parameter": "hidden_layer_sizes",
        "values": [(16,), (32,), (64,)],
    },
}


def format_value(value: object) -> str:
    return str(value)


def build_candidate_pipeline(model_name: str, param_value: object) -> Pipeline:
    dense_standard = build_preprocessor(RQ2_FEATURE_SET_B, scaler="standard", dense_output=True)
    dense_minmax = build_preprocessor(RQ2_FEATURE_SET_B, scaler="minmax", dense_output=True)

    if model_name == "Logistic Regression":
        classifier = LogisticRegression(
            C=float(param_value),
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        preprocessor = dense_standard
    elif model_name == "Linear SVM":
        classifier = LinearSVC(
            C=float(param_value),
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
        preprocessor = dense_standard
    elif model_name == "Multinomial Naive Bayes":
        classifier = MultinomialNB(alpha=float(param_value))
        preprocessor = dense_minmax
    elif model_name == "Decision Tree":
        classifier = DecisionTreeClassifier(
            max_depth=int(param_value),
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        preprocessor = dense_standard
    elif model_name == "Random Forest":
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=int(param_value),
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        preprocessor = dense_standard
    elif model_name == "MLP":
        classifier = MLPClassifier(
            hidden_layer_sizes=param_value,
            batch_size=512,
            early_stopping=True,
            learning_rate_init=0.0005,
            max_iter=150,
            random_state=RANDOM_STATE,
        )
        preprocessor = dense_standard
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_tuning_subset(X: pd.DataFrame, y: pd.Series, groups: pd.Series):
    # Keep tuning separate from the held-out test set
    split_indices = make_patient_aware_split(y, groups)

    X_train = X.iloc[split_indices.train_idx].reset_index(drop=True)
    y_train = y.iloc[split_indices.train_idx].reset_index(drop=True)
    groups_train = groups.iloc[split_indices.train_idx].reset_index(drop=True)

    subset_splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    _, subset_idx = next(subset_splitter.split(X_train, y_train, groups_train))

    return (
        X_train.iloc[subset_idx].reset_index(drop=True),
        y_train.iloc[subset_idx].reset_index(drop=True),
        groups_train.iloc[subset_idx].reset_index(drop=True),
    )


def run_tuning() -> tuple[pd.DataFrame, pd.DataFrame]:
    X, y, groups = load_modeling_frame(RQ2_FEATURE_SET_B)
    X_tune, y_tune, groups_tune = build_tuning_subset(X, y, groups)
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    tuning_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for model_name, config in TUNING_GRID.items():
        parameter = config["parameter"]
        best_value = None
        best_score = float("-inf")

        for value in config["values"]:
            pipeline = build_candidate_pipeline(model_name, value)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                warnings.simplefilter("ignore", category=UndefinedMetricWarning)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                scores = cross_validate(
                    pipeline,
                    X_tune,
                    y_tune,
                    groups=groups_tune,
                    cv=cv,
                    scoring=SCORING,
                    n_jobs=1,
                    error_score="raise",
                )

            row = {
                "model": model_name,
                "parameter": parameter,
                "value": format_value(value),
                "pr_auc_mean": scores["test_pr_auc"].mean(),
                "pr_auc_std": scores["test_pr_auc"].std(),
                "recall_mean": scores["test_recall_30d"].mean(),
                "f1_mean": scores["test_f1_30d"].mean(),
                "roc_auc_mean": scores["test_roc_auc"].mean(),
            }
            tuning_rows.append(row)

            if row["pr_auc_mean"] > best_score:
                best_score = row["pr_auc_mean"]
                best_value = value

        summary_rows.append(
            {
                "model": model_name,
                "parameter": parameter,
                "values_tried": ", ".join(format_value(v) for v in config["values"]),
                "selected_value": format_value(best_value),
                "selected_pr_auc_mean": best_score,
                "tuning_subset_n": len(X_tune),
                "tuning_subset_positive_rate": y_tune.mean(),
            }
        )

    tuning_results = pd.DataFrame(tuning_rows)
    tuning_summary = pd.DataFrame(summary_rows)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tuning_results.to_csv(TUNING_TABLE_PATH, index=False)
    tuning_summary.to_csv(TUNING_SUMMARY_PATH, index=False)

    return tuning_results, tuning_summary


def plot_tuning_curves(tuning_results: pd.DataFrame) -> None:
    model_order = list(TUNING_GRID.keys())
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.patch.set_facecolor("white")

    for ax, model_name in zip(axes.flatten(), model_order, strict=True):
        subset = tuning_results.loc[tuning_results["model"] == model_name].copy()
        subset["label"] = subset["value"].astype(str)
        ax.plot(subset["label"], subset["pr_auc_mean"], marker="o", color="#4C78A8")
        ax.fill_between(
            subset["label"],
            subset["pr_auc_mean"] - subset["pr_auc_std"],
            subset["pr_auc_mean"] + subset["pr_auc_std"],
            color="#4C78A8",
            alpha=0.15,
        )
        ax.set_title(model_name, fontsize=11)
        ax.set_ylabel("CV PR-AUC")
        ax.grid(axis="y", alpha=0.25)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle("Figure 1: Patient-aware tuning curves on the RQ2 Feature Set B subset", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(TUNING_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    tuning_results, tuning_summary = run_tuning()
    plot_tuning_curves(tuning_results)
    print(f"Wrote {TUNING_TABLE_PATH}")
    print(f"Wrote {TUNING_SUMMARY_PATH}")
    print(f"Wrote {TUNING_FIGURE_PATH}")
    print()
    print(tuning_summary.to_string(index=False))


if __name__ == "__main__":
    main()
