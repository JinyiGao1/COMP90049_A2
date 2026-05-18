# RQ1 Results Draft

## Cohort Definition

The original dataset contained 101,766 hospital encounters from 71,518 unique patients. For the main analysis, encounters with expired discharge dispositions were excluded because they are not clinically comparable to standard readmission-risk prediction cases. These were discharge disposition IDs 11, 19, 20, and 21. After this exclusion, the analysis cohort contained 100,114 encounters from 70,439 unique patients.

Hospice discharge encounters were retained in the main RQ1 cohort but should be mentioned as clinically non-comparable in the limitations or sensitivity discussion. The hospice-related IDs were 13 and 14, with 771 encounters in the raw data.

Each row represents a hospital encounter rather than a unique patient, so RQ1 should be described as an encounter-level analysis.

The cleaned CSV retains `encounter_id` for traceability and `patient_nbr` for later patient-aware splitting. Neither identifier should be used as a modelling feature.

## Outcome Distribution

After excluding expired discharges, the readmission outcome distribution was:

| Outcome | Encounters | Percentage |
|---|---:|---:|
| NO | 53,212 | 53.15% |
| >30 | 35,545 | 35.50% |
| <30 | 11,357 | 11.34% |

This confirms that `<30` readmission is a minority outcome, which motivates later model evaluation metrics beyond accuracy.

## Main Descriptive Patterns

The strongest descriptive difference is prior healthcare utilisation. Encounters followed by `<30` readmission had the highest proportion of prior inpatient utilisation: 49.76%, compared with 42.81% for `>30` and 23.63% for `NO`. Prior emergency use was also highest in `<30`: 16.64%, compared with 15.02% for `>30` and 7.44% for `NO`. Prior outpatient use was elevated in both readmitted groups but was slightly higher in `>30` than `<30`, so the report should not claim that every utilisation measure is highest for `<30`.

The `<30` group also showed higher encounter complexity. Median `num_medications` was 16 for `<30`, compared with 15 for `>30` and 14 for `NO`. Median `number_diagnoses` was 9 for both readmitted groups and 8 for `NO`. Median `time_in_hospital` was 4 days for both readmitted groups and 3 days for `NO`.

The generated Table 1-style numeric summary reports `n` and median (IQR) for the key continuous/count variables by readmission outcome. This table should be used as the main numerical support for the RQ1 written analysis, with plots used to highlight the most interpretable patterns.

Admission and discharge pathways also differed, but the admission-source pattern should be interpreted carefully. Both readmitted groups were more likely than `NO` encounters to come from the Emergency Room, so this pattern is better described as a readmitted-versus-not-readmitted difference rather than a uniquely `<30` pattern. Discharge disposition showed a clearer `<30` difference: transfer-related discharge destinations accounted for 30.00% of `<30` encounters, compared with 19.54% for `>30` and 19.53% for `NO`.

Diabetes management indicators showed smaller but relevant differences. Diabetes medication use was more common in `<30` encounters, and insulin adjustment levels were more frequent than in `NO` encounters. A1C result availability was limited, so A1C-related differences should be interpreted cautiously and should not be used as a major standalone claim.

Primary diagnosis patterns were directionally different but not dominant. Circulatory diagnoses were common in all groups: 30.69% for `<30`, 30.49% for `>30`, and 29.24% for `NO`. Diabetes as the primary diagnosis was slightly more common in `<30` encounters: 10.01%, compared with 9.33% for `>30` and 7.96% for `NO`. These coarse diagnosis groups support the clinical description without overcomplicating the RQ1 analysis.

## Statistical Evidence

The supporting statistical table reports Kruskal-Wallis tests for numeric variables and chi-square tests with Cramer's V for categorical variables. Because the dataset is large, p-values are often very small. The report should emphasise practical differences and effect sizes rather than treating statistical significance as the main finding.

The largest effect sizes were observed for prior utilisation, especially `number_inpatient`, and for discharge/admission pathway groups. Most categorical Cramer's V values were small, which is expected in a large heterogeneous clinical dataset.

## Suggested RQ1 Conclusion

The RQ1 analysis suggests that encounters followed by readmission within 30 days differ from later-readmission and no-readmission encounters in several clinically relevant ways. In particular, the `<30` group shows higher prior inpatient and emergency utilisation, greater treatment or diagnosis complexity, and different discharge pathway patterns. These descriptive differences do not prove that early readmission can be predicted accurately, but they provide empirical motivation for treating `<30` readmission as the positive high-risk class in the subsequent modelling task.

## Generated Outputs

- `reports/figures/rq1_figure_01_outcome_distribution_post_expired_exclusion.png`
- `reports/figures/rq1_figure_02_prior_utilisation_post_expired_exclusion.png`
- `reports/figures/rq1_figure_03_admission_source_post_expired_exclusion.png`
- `reports/figures/rq1_figure_04_discharge_disposition_post_expired_exclusion.png`
- `reports/figures/rq1_figure_05_diabetes_management_post_expired_exclusion.png`
- `reports/tables/rq1_table_01_numeric_summary_by_readmitted_post_expired_exclusion.csv`
- `reports/tables/rq1_table_04_discharge_disposition_percentages_post_expired_exclusion.csv`
- `reports/tables/rq1_table_support_diag_1_group_percentages_post_expired_exclusion.csv`
- `reports/tables/rq1_table_support_key_test_summary_post_expired_exclusion.csv`
