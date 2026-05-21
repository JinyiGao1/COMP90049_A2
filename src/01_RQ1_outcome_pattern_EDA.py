# %% [markdown]
# # RQ1 Outcome Pattern EDA
#
# Research question:
#
# Do encounters followed by readmission within 30 days show different
# encounter-level patterns from later-readmission and no-readmission encounters?
#
# This notebook-style script is intentionally focused on descriptive evidence.
# Statistical tests are generated as supporting evidence, not as the main story.

# %%
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from data_cleaning import write_outputs as write_cleaned_outputs
from rq1_eda import write_outputs as write_rq1_outputs

TABLE_DIR = PROJECT_ROOT / "reports" / "tables"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"


# %% [markdown]
# ## 1. Build the shared cleaned cohort
#
# Cleaning decisions:
#
# - Replace `?` with missing values.
# - Exclude expired discharge encounters (`11`, `19`, `20`, `21`).
# - Retain hospice encounters for the main analysis, but report their count.
# - Drop `weight` and `payer_code` from the cleaned feature table due to high missingness.
# - Keep `encounter_id` only for traceability.
# - Keep `patient_nbr` for later patient-aware splitting, but do not treat rows as unique patients.
# - Exclude both `encounter_id` and `patient_nbr` from model features later.
# - Add compact grouped features for admission source/type, discharge destination, diagnosis category,
#   prior healthcare utilisation, and diabetes medication burden.

# %%
write_cleaned_outputs()

# %%
pd.read_csv(TABLE_DIR / "cohort_summary.csv")

# %%
pd.read_csv(TABLE_DIR / "expired_hospice_discharge_summary.csv")

# %%
pd.read_csv(TABLE_DIR / "raw_missing_summary.csv")

# %%
pd.read_csv(TABLE_DIR / "model_feature_exclusion_notes.csv")


# %% [markdown]
# ## 2. Generate RQ1 tables and figures
#
# The generated outputs are designed for a compact report section:
#
# - Figure 1: readmission outcome distribution.
# - Table 1: numeric descriptors by outcome group.
# - Figure 2: prior utilisation indicators.
# - Figure 3: admission source pattern.
# - Figure 4: discharge disposition pattern.
# - Figure 5: diabetes management indicators.
# - Supporting table: limited statistical tests and effect sizes.

# %%
write_rq1_outputs()

# %%
pd.read_csv(TABLE_DIR / "rq1_table_00_outcome_distribution_post_expired_exclusion.csv")

# %%
pd.read_csv(TABLE_DIR / "rq1_table_01_numeric_summary_by_readmitted_post_expired_exclusion.csv")

# %%
pd.read_csv(TABLE_DIR / "rq1_table_02_prior_utilisation_rates_post_expired_exclusion.csv")

# %%
pd.read_csv(TABLE_DIR / "rq1_table_03_admission_source_percentages_post_expired_exclusion.csv")

# %%
pd.read_csv(TABLE_DIR / "rq1_table_04_discharge_disposition_percentages_post_expired_exclusion.csv")

# %%
pd.read_csv(TABLE_DIR / "rq1_table_05_diabetes_management_percentages_post_expired_exclusion.csv")

# %%
pd.read_csv(TABLE_DIR / "rq1_table_support_diag_1_group_percentages_post_expired_exclusion.csv")

# %%
pd.read_csv(TABLE_DIR / "rq1_table_support_key_test_summary_post_expired_exclusion.csv")


# %% [markdown]
# ## 3. Interpretation notes
#
# Use "encounters" rather than "patients" when describing RQ1, because the data are encounter-level
# and some patients appear more than once.
#
# Good wording:
#
# > Encounters followed by `<30` readmission had higher prior inpatient and emergency utilisation,
# > a higher proportion of transfer-related discharge destinations, and slightly greater treatment
# > or diagnostic complexity than encounters with later readmission or no recorded readmission.
#
# Avoid overclaiming:
#
# > RQ1 does not prove that `<30` can be predicted accurately. It provides empirical motivation for
# > treating `<30` as the positive high-risk outcome in subsequent modelling.
