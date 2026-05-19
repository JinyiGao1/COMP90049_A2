from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_RESULTS_PATH = PROJECT_ROOT / "results" / "tables" / "rq2_baseline_test_results.csv"
DELTA_RESULTS_PATH = PROJECT_ROOT / "results" / "tables" / "rq2_baseline_delta_results.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "figures" / "rq2_figure_01_feature_set_comparison.png"

MODEL_ORDER = [
    "Logistic Regression",
    "Linear SVM",
    "Decision Tree",
    "Random Forest",
    "Naive Bayes",
    "MLP",
]

MODEL_LABELS = {
    "Logistic Regression": "LogReg",
    "Linear SVM": "Linear SVM",
    "Decision Tree": "Tree",
    "Random Forest": "Forest",
    "Naive Bayes": "Naive Bayes",
    "MLP": "MLP",
}


def read_test_pr_auc(path: Path) -> dict[tuple[str, str], float]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        (row["model"], row["feature_set"]): float(row["pr_auc"])
        for row in rows
    }


def read_delta_pr_auc(path: Path) -> dict[str, float]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        row["model"]: float(row["delta_pr_auc"])
        for row in rows
    }


def plot_rq2_feature_comparison(
    pr_auc: dict[tuple[str, str], float],
    deltas: dict[str, float],
    output_path: Path,
    title: str = "RQ2: Incremental value of glycaemic-management features",
) -> None:
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.0),
        gridspec_kw={"width_ratios": [1.35, 1]},
    )
    fig.patch.set_facecolor("white")

    ax = axes[0]
    x_positions = list(range(len(MODEL_ORDER)))
    width = 0.36
    ax.bar(
        [x - width / 2 for x in x_positions],
        [pr_auc[(model, "A")] for model in MODEL_ORDER],
        width=width,
        label="A: baseline",
        color="#4C78A8",
    )
    ax.bar(
        [x + width / 2 for x in x_positions],
        [pr_auc[(model, "B")] for model in MODEL_ORDER],
        width=width,
        label="B: + diabetes management",
        color="#F58518",
    )
    ax.set_title("Test PR-AUC by feature set", fontsize=12, pad=10)
    ax.set_ylabel("PR-AUC")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        rotation=25,
        ha="right",
    )
    ax.set_ylim(0, 0.225)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.62, 0.99))
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    ordered_models = sorted(MODEL_ORDER, key=lambda model: deltas[model])
    delta_values = [deltas[model] for model in ordered_models]
    labels = [MODEL_LABELS[model] for model in ordered_models]
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in delta_values]

    ax.barh(labels, delta_values, color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlim(-0.006, 0.050)
    ax.set_title("Change in PR-AUC after adding diabetes features", fontsize=12, pad=10)
    ax.set_xlabel("Delta PR-AUC (B - A)")
    ax.grid(axis="x", alpha=0.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for index, value in enumerate(delta_values):
        label_x = value + 0.0012 if value >= 0 else 0.0012
        ax.text(label_x, index, f"{value:+.3f}", va="center", ha="left", fontsize=9)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout(w_pad=3.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    pr_auc = read_test_pr_auc(TEST_RESULTS_PATH)
    deltas = read_delta_pr_auc(DELTA_RESULTS_PATH)
    plot_rq2_feature_comparison(pr_auc, deltas, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
