from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).resolve().parents[1] / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_diabetic_encounters.csv"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
TABLE_DIR = PROJECT_ROOT / "reports" / "tables"

READMITTED_ORDER = ["NO", ">30", "<30"]
POST_EXCLUSION_SUFFIX = "post_expired_exclusion"

NUMERIC_RQ1_COLUMNS = [
    "time_in_hospital",
    "num_medications",
    "number_diagnoses",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "prior_utilisation",
    "active_diabetes_med_count",
    "changed_diabetes_med_count",
]

TABLE1_NUMERIC_COLUMNS = [
    "time_in_hospital",
    "num_medications",
    "number_diagnoses",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
]

CATEGORICAL_RQ1_COLUMNS = [
    "admission_type_group",
    "admission_source_group",
    "discharge_disposition_group",
    "A1Cresult",
    "max_glu_serum",
    "change",
    "diabetesMed",
    "insulin",
    "diag_1_group",
    "has_diabetes_diag",
]


def ordered_readmitted(series: pd.Series) -> pd.Series:
    return pd.Categorical(series, categories=READMITTED_ORDER, ordered=True)


def cramers_v(table: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    r, k = table.shape
    denominator = n * (min(k - 1, r - 1))
    return float(np.sqrt(chi2 / denominator)) if denominator else np.nan


def kruskal_effect_size(groups: list[pd.Series], h_stat: float) -> float:
    n = sum(len(group) for group in groups)
    k = len(groups)
    if n <= k:
        return np.nan
    return float(max((h_stat - k + 1) / (n - k), 0))


def summarise_numeric(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in NUMERIC_RQ1_COLUMNS:
        for outcome in READMITTED_ORDER:
            values = df.loc[df["readmitted"] == outcome, column].dropna()
            rows.append(
                {
                    "variable": column,
                    "readmitted": outcome,
                    "n": len(values),
                    "mean": values.mean(),
                    "median": values.median(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                    "iqr": values.quantile(0.75) - values.quantile(0.25),
                }
            )
    return pd.DataFrame(rows)


def build_table1_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "measure": "n encounters",
            **{
                outcome: f"{int((df['readmitted'] == outcome).sum()):,}"
                for outcome in READMITTED_ORDER
            },
        }
    ]

    for column in TABLE1_NUMERIC_COLUMNS:
        row = {"measure": column}
        for outcome in READMITTED_ORDER:
            values = df.loc[df["readmitted"] == outcome, column].dropna()
            median = values.median()
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            row[outcome] = f"{median:.0f} ({q1:.0f}-{q3:.0f})"
        rows.append(row)

    return pd.DataFrame(rows)


def summarise_categorical(df: pd.DataFrame, column: str) -> pd.DataFrame:
    table = (
        pd.crosstab(df[column].fillna("Missing"), df["readmitted"], normalize="columns")
        .reindex(columns=READMITTED_ORDER)
        .mul(100)
        .round(2)
    )
    return table.rename_axis(index=column).reset_index()


def build_test_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in NUMERIC_RQ1_COLUMNS:
        groups = [df.loc[df["readmitted"] == outcome, column].dropna() for outcome in READMITTED_ORDER]
        h_stat, p_value = stats.kruskal(*groups)
        rows.append(
            {
                "variable": column,
                "type": "numeric",
                "test": "Kruskal-Wallis",
                "statistic": h_stat,
                "p_value": p_value,
                "effect_size": kruskal_effect_size(groups, h_stat),
                "effect_size_name": "epsilon_squared",
            }
        )

    for column in CATEGORICAL_RQ1_COLUMNS:
        table = pd.crosstab(df[column].fillna("Missing"), df["readmitted"])
        chi2, p_value, _, _ = stats.chi2_contingency(table)
        rows.append(
            {
                "variable": column,
                "type": "categorical",
                "test": "Chi-square",
                "statistic": chi2,
                "p_value": p_value,
                "effect_size": cramers_v(table),
                "effect_size_name": "cramers_v",
            }
        )

    return pd.DataFrame(rows).sort_values(["type", "effect_size"], ascending=[True, False])


def save_outcome_distribution(df: pd.DataFrame) -> None:
    distribution = (
        df["readmitted"]
        .value_counts()
        .reindex(READMITTED_ORDER)
        .rename_axis("readmitted")
        .reset_index(name="count")
    )
    distribution["percent"] = distribution["count"] / distribution["count"].sum() * 100
    distribution.to_csv(
        TABLE_DIR / f"rq1_table_00_outcome_distribution_{POST_EXCLUSION_SUFFIX}.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(data=distribution, x="readmitted", y="percent", order=READMITTED_ORDER, ax=ax)
    ax.set_xlabel("Readmission outcome")
    ax.set_ylabel("Encounters (%)")
    ax.set_title("Readmission outcome distribution")
    ax.set_ylim(0, distribution["percent"].max() + 8)
    ax.bar_label(ax.containers[0], labels=[f"{v:.1f}%" for v in distribution["percent"]], padding=3)
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / f"rq1_figure_01_outcome_distribution_{POST_EXCLUSION_SUFFIX}.png",
        dpi=200,
    )
    plt.close(fig)


def save_prior_utilisation_plot(df: pd.DataFrame) -> None:
    rate_df = (
        df.groupby("readmitted", observed=False)[
            ["has_prior_outpatient", "has_prior_emergency", "has_prior_inpatient"]
        ]
        .mean()
        .reindex(READMITTED_ORDER)
        .mul(100)
        .reset_index()
        .melt(id_vars="readmitted", var_name="prior_use_type", value_name="percent")
    )
    labels = {
        "has_prior_outpatient": "Any prior outpatient",
        "has_prior_emergency": "Any prior emergency",
        "has_prior_inpatient": "Any prior inpatient",
    }
    rate_df["prior_use_type"] = rate_df["prior_use_type"].map(labels)
    rate_df.to_csv(
        TABLE_DIR / f"rq1_table_02_prior_utilisation_rates_{POST_EXCLUSION_SUFFIX}.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(
        data=rate_df,
        x="prior_use_type",
        y="percent",
        hue="readmitted",
        hue_order=READMITTED_ORDER,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Encounters (%)")
    ax.set_title("Prior healthcare utilisation by readmission outcome")
    ax.legend(title="Readmitted")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / f"rq1_figure_02_prior_utilisation_{POST_EXCLUSION_SUFFIX}.png",
        dpi=200,
    )
    plt.close(fig)


def save_admission_source_plot(df: pd.DataFrame) -> None:
    pathway = summarise_categorical(df, "admission_source_group")
    pathway_long = pathway.melt(
        id_vars="admission_source_group", var_name="readmitted", value_name="percent"
    )
    pathway_long.to_csv(
        TABLE_DIR / f"rq1_table_03_admission_source_percentages_{POST_EXCLUSION_SUFFIX}.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.barplot(
        data=pathway_long,
        x="admission_source_group",
        y="percent",
        hue="readmitted",
        hue_order=READMITTED_ORDER,
        ax=ax,
    )
    ax.set_xlabel("Admission source group")
    ax.set_ylabel("Within-outcome encounters (%)")
    ax.set_title("Admission source pattern by readmission outcome")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Readmitted")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / f"rq1_figure_03_admission_source_{POST_EXCLUSION_SUFFIX}.png",
        dpi=200,
    )
    plt.close(fig)


def save_discharge_disposition_plot(df: pd.DataFrame) -> None:
    discharge = summarise_categorical(df, "discharge_disposition_group")
    discharge_long = discharge.melt(
        id_vars="discharge_disposition_group", var_name="readmitted", value_name="percent"
    )
    discharge_long.to_csv(
        TABLE_DIR / f"rq1_table_04_discharge_disposition_percentages_{POST_EXCLUSION_SUFFIX}.csv",
        index=False,
    )

    order = ["Home", "Home health/IV", "Transferred facility", "Hospice", "Other/Unknown"]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=discharge_long,
        x="discharge_disposition_group",
        y="percent",
        order=order,
        hue="readmitted",
        hue_order=READMITTED_ORDER,
        ax=ax,
    )
    ax.set_xlabel("Discharge disposition group")
    ax.set_ylabel("Within-outcome encounters (%)")
    ax.set_title("Discharge disposition pattern by readmission outcome")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="Readmitted")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / f"rq1_figure_04_discharge_disposition_{POST_EXCLUSION_SUFFIX}.png",
        dpi=200,
    )
    plt.close(fig)


def save_diabetes_management_plot(df: pd.DataFrame) -> None:
    diabetes_columns = ["A1Cresult", "change", "diabetesMed", "insulin"]
    parts = []
    for column in diabetes_columns:
        table = summarise_categorical(df, column)
        long = table.melt(id_vars=column, var_name="readmitted", value_name="percent")
        long = long.rename(columns={column: "category"})
        long["variable"] = column
        parts.append(long)

    diabetes_summary = pd.concat(parts, ignore_index=True)
    diabetes_summary.to_csv(
        TABLE_DIR / f"rq1_table_05_diabetes_management_percentages_{POST_EXCLUSION_SUFFIX}.csv",
        index=False,
    )

    selected = diabetes_summary[
        (
            (diabetes_summary["variable"] == "A1Cresult")
            & (diabetes_summary["category"].isin([">7", ">8", "Norm"]))
        )
        | ((diabetes_summary["variable"] == "change") & (diabetes_summary["category"] == "Ch"))
        | ((diabetes_summary["variable"] == "diabetesMed") & (diabetes_summary["category"] == "Yes"))
        | ((diabetes_summary["variable"] == "insulin") & (diabetes_summary["category"].isin(["Up", "Down", "Steady"])))
    ].copy()
    selected["feature_level"] = selected["variable"] + ": " + selected["category"].astype(str)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=False)
    axis_map = {
        "A1Cresult": axes[0, 0],
        "change": axes[0, 1],
        "diabetesMed": axes[1, 0],
        "insulin": axes[1, 1],
    }
    titles = {
        "A1Cresult": "A1C result",
        "change": "Medication change",
        "diabetesMed": "Diabetes medication",
        "insulin": "Insulin dosage",
    }
    category_orders = {
        "A1Cresult": ["Norm", ">7", ">8"],
        "change": ["Ch"],
        "diabetesMed": ["Yes"],
        "insulin": ["Down", "Steady", "Up"],
    }

    for variable, ax in axis_map.items():
        subset = selected[selected["variable"] == variable].copy()
        sns.barplot(
            data=subset,
            x="category",
            y="percent",
            order=category_orders[variable],
            hue="readmitted",
            hue_order=READMITTED_ORDER,
            ax=ax,
        )
        ax.set_title(titles[variable])
        ax.set_xlabel("")
        ax.set_ylabel("Encounters (%)")
        ax.legend_.remove()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Diabetes management indicators by readmission outcome", y=0.99)
    fig.legend(
        handles,
        labels,
        title="Readmitted",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(
        FIGURE_DIR / f"rq1_figure_05_diabetes_management_{POST_EXCLUSION_SUFFIX}.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_summary_note(df: pd.DataFrame, numeric_summary: pd.DataFrame) -> None:
    dist = (
        df["readmitted"].value_counts().reindex(READMITTED_ORDER).rename_axis("readmitted").to_frame("count")
    )
    dist["percent"] = dist["count"] / len(df) * 100

    key_medians = numeric_summary[
        numeric_summary["variable"].isin(
            ["number_inpatient", "number_emergency", "number_outpatient", "num_medications", "number_diagnoses"]
        )
    ].copy()

    note = [
        "# RQ1 Working Summary",
        "",
        "Each row is treated as a hospital encounter. The EDA therefore compares encounter-level patterns, not fully independent unique patients.",
        "",
        "Main cohort definition: expired discharges were excluded; hospice discharges were retained and can be discussed as clinically non-comparable in limitations or sensitivity analysis.",
        "",
        "`encounter_id` is retained only for traceability. `patient_nbr` is retained for later patient-aware splitting. Both identifiers should be excluded from model features.",
        "",
        "## Outcome Distribution",
        "",
        dist.round({"percent": 2}).to_markdown(),
        "",
        "## Key Numeric Medians",
        "",
        key_medians[["variable", "readmitted", "median", "q1", "q3"]].round(2).to_markdown(index=False),
        "",
        "## Suggested RQ1 Conclusion Wording",
        "",
        "The RQ1 analysis suggests that encounters followed by readmission within 30 days differ from later-readmission and no-readmission encounters in several clinically relevant ways. In particular, the <30 group shows higher prior inpatient and emergency utilisation, greater treatment or diagnosis complexity, and different discharge pathway patterns. These descriptive differences do not prove that early readmission can be predicted accurately, but they provide empirical motivation for treating <30 readmission as the positive high-risk class in the subsequent modelling task.",
        "",
    ]
    (TABLE_DIR / f"rq1_working_summary_{POST_EXCLUSION_SUFFIX}.md").write_text(
        "\n".join(note), encoding="utf-8"
    )


def write_outputs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["readmitted"] = ordered_readmitted(df["readmitted"])

    numeric_summary = summarise_numeric(df)
    numeric_summary.to_csv(
        TABLE_DIR / f"rq1_table_01a_numeric_summary_long_{POST_EXCLUSION_SUFFIX}.csv",
        index=False,
    )
    table1 = build_table1_numeric_summary(df)
    table1.to_csv(
        TABLE_DIR / f"rq1_table_01_numeric_summary_by_readmitted_{POST_EXCLUSION_SUFFIX}.csv",
        index=False,
    )

    for column in CATEGORICAL_RQ1_COLUMNS:
        summarise_categorical(df, column).to_csv(
            TABLE_DIR / f"rq1_table_support_{column}_percentages_{POST_EXCLUSION_SUFFIX}.csv",
            index=False,
        )

    test_summary = build_test_summary(df)
    test_summary.to_csv(
        TABLE_DIR / f"rq1_table_support_key_test_summary_{POST_EXCLUSION_SUFFIX}.csv",
        index=False,
    )

    save_outcome_distribution(df)
    save_prior_utilisation_plot(df)
    save_admission_source_plot(df)
    save_discharge_disposition_plot(df)
    save_diabetes_management_plot(df)
    write_summary_note(df, numeric_summary)

    print(f"Wrote RQ1 tables to {TABLE_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Wrote RQ1 figures to {FIGURE_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    write_outputs()
