# Imports

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold, cross_validate


# Import Logistic Regression model
from sklearn.linear_model import LogisticRegression

# Import Naive Bayes model
from sklearn.naive_bayes import MultinomialNB

# Import MLP Neural Netword model
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

# Import Linear SVM
from sklearn.svm import LinearSVC

# Import Decision tree model from sklearn
from sklearn.tree import DecisionTreeClassifier

from src.paths import DATA_DIR, TABLES_DIR

# Fix random seed to make results reproducible
RANDOM_STATE = 42

# Define target column, group column, identifier columns, and data path
TARGET_COLUMN = "early_readmission"
GROUP_COLUMN = "patient_nbr"
IDENTIFIER_COLUMNS = ["encounter_id", "patient_nbr", "readmitted", TARGET_COLUMN]
DATA_PATH = DATA_DIR / "processed" / "cleaned_diabetic_encounters.csv"




# ------------- Define feature sets and modeling utilities ------------- #
# RQ2: general baseline feature set
RQ2_FEATURE_SET_A = [
    "age",
    "gender",
    "race",
    "admission_type_group",
    "admission_source_group",
    "discharge_disposition_group",
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "prior_utilisation",
    "has_prior_outpatient",
    "has_prior_emergency",
    "has_prior_inpatient",
    "diag_1_group",
    "has_diabetes_diag",
]

# RQ2: general baseline feature set + diabetes management features
RQ2_FEATURE_SET_B = RQ2_FEATURE_SET_A + [
    "A1Cresult",
    "max_glu_serum",
    "change",
    "diabetesMed",
    "insulin",
    "active_diabetes_med_count",
    "changed_diabetes_med_count",
]

# Run the same models on A and B
RQ2_FEATURE_SETS = {
    "A": RQ2_FEATURE_SET_A,
    "B": RQ2_FEATURE_SET_B,
}

# Separarte columns
NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "prior_utilisation",
    "active_diabetes_med_count",
    "changed_diabetes_med_count",
]

BOOLEAN_FEATURES = [
    "has_prior_outpatient",
    "has_prior_emergency",
    "has_prior_inpatient",
    "has_diabetes_diag",
]

# for cross-validation
SCORING = {
    "precision_30d": "precision",
    "recall_30d": "recall",
    "f1_30d": "f1",
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
}


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    groups_train: pd.Series
    groups_test: pd.Series
    split_method: str


@dataclass(frozen=True)
class SplitIndices:
    train_idx: list[int]
    test_idx: list[int]
    split_method: str

# --------------------- Preprocessing ----------------- #

# Convert variables to numbers e.g. male -> 0, female -> 1, unknown -> 2, other -> 3
def build_one_hot_encoder(dense_output: bool) -> OneHotEncoder:
    kwargs = {"handle_unknown": "ignore"}
    try:
        return OneHotEncoder(sparse_output=not dense_output, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=not dense_output, **kwargs)


# 1. find numeric features 
# 2. find categorical features
# 3. process numeric and categorical features differently
# - numeric: impute missing values with median, then scale (standard or minmax)
# - categorical: impute missing values with most frequent, then one-hot encode

def build_preprocessor(feature_columns: list[str], scaler: str = "standard", dense_output: bool = True):
    numeric_features = [column for column in NUMERIC_FEATURES if column in feature_columns]
    boolean_features = [column for column in BOOLEAN_FEATURES if column in feature_columns]
    numeric_features = numeric_features + boolean_features
    categorical_features = [
        column
        for column in feature_columns
        if column not in numeric_features
    ]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaler == "standard":
        numeric_steps.append(("scaler", StandardScaler()))
    elif scaler == "minmax":
        numeric_steps.append(("scaler", MinMaxScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", build_one_hot_encoder(dense_output=dense_output)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

# Read data, check for missing columns, and return X, y, and groups
def load_modeling_frame(feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    missing_columns = sorted(set(feature_columns + [TARGET_COLUMN, GROUP_COLUMN]) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    groups = df[GROUP_COLUMN].copy()
    return X, y, groups


# ---------------- Prevent Data Leaks ---------------- #
# Train/test split that is patient-aware (i.e. no patient appears in both train and test) and stratified on the target variable. 
# If StratifiedGroupKFold fails (e.g. due to too few samples in some groups), fall back to GroupShuffleSplit which is not stratified but still patient-aware.
def make_patient_aware_split(
    y: pd.Series,
    groups: pd.Series,
) -> SplitIndices:
    dummy_X = pd.DataFrame(index=range(len(y)))
    try:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(dummy_X, y, groups))
        split_method = "StratifiedGroupKFold first split"
    except Exception:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(dummy_X, y, groups))
        split_method = "GroupShuffleSplit fallback"

    return SplitIndices(
        train_idx=train_idx.tolist(),
        test_idx=test_idx.tolist(),
        split_method=split_method,
    )

# cross-validation splitter
def build_cv_splitter():
    try:
        return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), "StratifiedGroupKFold"
    except Exception:
        return GroupKFold(n_splits=5), "GroupKFold fallback"

# apply the same split indices to the current feature set, as A columns and b columns are different, but rows are the same
def build_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    split_indices: SplitIndices,
) -> SplitData:
    # Use the same split for A and B
    return SplitData(
        X_train=X.iloc[split_indices.train_idx].reset_index(drop=True),
        X_test=X.iloc[split_indices.test_idx].reset_index(drop=True),
        y_train=y.iloc[split_indices.train_idx].reset_index(drop=True),
        y_test=y.iloc[split_indices.test_idx].reset_index(drop=True),
        groups_train=groups.iloc[split_indices.train_idx].reset_index(drop=True),
        groups_test=groups.iloc[split_indices.test_idx].reset_index(drop=True),
        split_method=split_indices.split_method,
    )

# Build pipelines for each model, with the same preprocessing steps for each feature set.
def build_model_pipelines(feature_columns: list[str]) -> dict[str, Pipeline]:
    dense_standard = build_preprocessor(feature_columns, scaler="standard", dense_output=True)
    dense_minmax = build_preprocessor(feature_columns, scaler="minmax", dense_output=True)

    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", dense_standard),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        solver="liblinear",
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Naive Bayes": Pipeline(
            steps=[
                ("preprocessor", dense_minmax),
                ("classifier", MultinomialNB()),
            ]
        ),
        "Decision Tree": Pipeline(
            steps=[
                ("preprocessor", dense_standard),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=12,
                        min_samples_leaf=20,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", dense_standard),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        min_samples_leaf=10,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Linear SVM": Pipeline(
            steps=[
                ("preprocessor", dense_standard),
                (
                    "classifier",
                    LinearSVC(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "MLP": Pipeline(
            steps=[
                ("preprocessor", dense_standard),
                (
                    "classifier",
                    MLPClassifier(
                        hidden_layer_sizes=(32,),
                        batch_size=512,
                        early_stopping=True,
                        learning_rate_init=0.0005,
                        max_iter=150,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


# train a model and then evaluate on the test set
def evaluate_on_test_set(model_name: str, pipeline: Pipeline, split_data: SplitData) -> dict[str, float | str]:
    pipeline.fit(split_data.X_train, split_data.y_train)
    y_pred = pipeline.predict(split_data.X_test)
    y_score = None

    if hasattr(pipeline, "predict_proba"):
        y_score = pipeline.predict_proba(split_data.X_test)[:, 1]
    elif hasattr(pipeline, "decision_function"):
        y_score = pipeline.decision_function(split_data.X_test)

    row: dict[str, float | str] = {
        "model": model_name,
        "precision_30d": precision_score(split_data.y_test, y_pred, zero_division=0),
        "recall_30d": recall_score(split_data.y_test, y_pred, zero_division=0),
        "f1_30d": f1_score(split_data.y_test, y_pred, zero_division=0),
        "test_positive_rate": float(split_data.y_test.mean()),
    }

    if y_score is not None:
        row["pr_auc"] = average_precision_score(split_data.y_test, y_score)
        row["roc_auc"] = roc_auc_score(split_data.y_test, y_score)

    return row


# --------------------- Delta Test ----------------- #
# Feature Set B result - Feature Set A result
# if result is positive, then diabetes management features helped
# if result is negative, then diabetes management features did not help, or may add noise
def build_delta_table(test_results: pd.DataFrame) -> pd.DataFrame:
    # Measure what B adds beyond A
    pivot = test_results.pivot(index="model", columns="feature_set")
    delta_table = pd.DataFrame(
        {
            "model": pivot.index,
            "delta_pr_auc": pivot[("pr_auc", "B")] - pivot[("pr_auc", "A")],
            "delta_recall_30d": pivot[("recall_30d", "B")] - pivot[("recall_30d", "A")],
            "delta_f1_30d": pivot[("f1_30d", "B")] - pivot[("f1_30d", "A")],
        }
    )
    return delta_table.sort_values("delta_pr_auc", ascending=False).reset_index(drop=True)



# --------------------- Run The Model Comparison ----------------- #
# 1. 默认使用 RQ2_FEATURE_SETS，也就是 A 和 B
# 2. 读取 longest feature set，用来拿 y 和 groups
# 3. 先创建一次 patient-aware split
# 4. 对 A 和 B 分别跑同样的 models
# 5. 做 cross-validation
# 6. 做 test set evaluation
# 7. 生成 delta table
# 8. 保存所有 CSV
def run_rq2_baseline(
    feature_sets: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_sets = feature_sets or RQ2_FEATURE_SETS
    base_feature_columns = max(feature_sets.values(), key=len)
    _, y, groups = load_modeling_frame(base_feature_columns)
    split_indices = make_patient_aware_split(y, groups)
    cv_splitter, cv_name = build_cv_splitter()

    cv_rows: list[dict[str, float | str]] = []
    test_rows: list[dict[str, float | str]] = []

    for feature_set_name, feature_columns in feature_sets.items():
        # Keep the whole experiment the same except the feature set
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

    # Table 1: CV performance
    cv_results = (
        pd.DataFrame(cv_rows)
        .sort_values(["feature_set", "pr_auc_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # Table 2: Test performance
    test_results = (
        pd.DataFrame(test_rows)[
            ["model", "feature_set", "pr_auc", "recall_30d", "precision_30d", "f1_30d", "roc_auc"]
        ]
        .sort_values(["feature_set", "pr_auc"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # Table 3: Delta table
    delta_table = build_delta_table(test_results)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(TABLES_DIR / "rq2_baseline_cv_results.csv", index=False)
    test_results.to_csv(TABLES_DIR / "rq2_baseline_test_results.csv", index=False)
    delta_table.to_csv(TABLES_DIR / "rq2_baseline_delta_results.csv", index=False)
    split_summary.to_csv(TABLES_DIR / "rq2_baseline_split_summary.csv", index=False)

    return cv_results, test_results, delta_table, split_summary


if __name__ == "__main__":
    cv_results, test_results, delta_table, split_summary = run_rq2_baseline()
    print("Wrote results/tables/rq2_baseline_cv_results.csv")
    print("Wrote results/tables/rq2_baseline_test_results.csv")
    print("Wrote results/tables/rq2_baseline_delta_results.csv")
    print("Wrote results/tables/rq2_baseline_split_summary.csv")
    print()
    print(split_summary.to_string(index=False))
    print()
    print(cv_results.to_string(index=False))
    print()
    print(delta_table.to_string(index=False))
