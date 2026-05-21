# COMP90049 Assignment 2: Diabetes Readmission Prediction

This repository contains the group project for COMP90049 Introduction to Machine Learning, Semester 1 2026.

The project studies short-term hospital readmission risk using the Diabetes 130-US Hospitals dataset. The main predictive task is to identify encounters that lead to readmission within 30 days.

## Research Questions

RQ1: Do early readmission, late readmission, and no readmission show different clinical, medication, and healthcare-utilisation patterns?

RQ2: Do glycaemic-management features, including HbA1c testing and medication changes, improve early-readmission prediction beyond general patient complexity and prior healthcare-utilisation features?

RQ3: Are model performance and false-negative patterns consistent across clinically relevant subgroups, such as age bands, race, and admission pathway?

## Repository Structure

```text
data/raw/                 Original dataset files
src/                      Reusable project code
reports/figures/          Generated report plots
results/tables/           Generated result tables
reports/                  Result drafts, figures, and report-ready outputs
report/                   ACL report template and final PDF
docs/                     Assignment specification, rubric, and group contract
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you prefer Anaconda, create an environment with Python 3.10 or newer and install the packages in `requirements.txt`.

## Data

The dataset files are placed in:

```text
data/raw/diabetic_data.csv
data/raw/IDS_mapping.csv
```

Dataset source: UCI Machine Learning Repository, Diabetes 130-US Hospitals for Years 1999-2008.

## Reproducing Main Outputs

Run the project scripts from the repository root after installing the dependencies:

```bash
PYTHONPATH=. python -m src.rq2_baseline
python src.rq2_visualization.py
python src.rq2_sensitivity_no_discharge.py
python src/rq3_subgroup_analysis.py
```

## Collaboration Notes

- Keep generated figures in `reports/figures/`.
- Keep generated result tables in `results/tables/`.
- Use clear commit messages describing the experiment or report section changed.
- The final submitted code should reproduce the main tables and figures used in the report.
