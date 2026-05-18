# RQ1 Working Summary

Each row is treated as a hospital encounter. The EDA therefore compares encounter-level patterns, not fully independent unique patients.

Main cohort definition: expired discharges were excluded; hospice discharges were retained and can be discussed as clinically non-comparable in limitations or sensitivity analysis.

`encounter_id` is retained only for traceability. `patient_nbr` is retained for later patient-aware splitting. Both identifiers should be excluded from model features.

## Outcome Distribution

| readmitted   |   count |   percent |
|:-------------|--------:|----------:|
| NO           |   53212 |     53.15 |
| >30          |   35545 |     35.5  |
| <30          |   11357 |     11.34 |

## Key Numeric Medians

| variable          | readmitted   |   median |   q1 |   q3 |
|:------------------|:-------------|---------:|-----:|-----:|
| num_medications   | NO           |       14 |   10 |   20 |
| num_medications   | >30          |       15 |   11 |   20 |
| num_medications   | <30          |       16 |   11 |   21 |
| number_diagnoses  | NO           |        8 |    5 |    9 |
| number_diagnoses  | >30          |        9 |    6 |    9 |
| number_diagnoses  | <30          |        9 |    6 |    9 |
| number_outpatient | NO           |        0 |    0 |    0 |
| number_outpatient | >30          |        0 |    0 |    0 |
| number_outpatient | <30          |        0 |    0 |    0 |
| number_emergency  | NO           |        0 |    0 |    0 |
| number_emergency  | >30          |        0 |    0 |    0 |
| number_emergency  | <30          |        0 |    0 |    0 |
| number_inpatient  | NO           |        0 |    0 |    0 |
| number_inpatient  | >30          |        0 |    0 |    1 |
| number_inpatient  | <30          |        0 |    0 |    2 |

## Suggested RQ1 Conclusion Wording

The RQ1 analysis suggests that encounters followed by readmission within 30 days differ from later-readmission and no-readmission encounters in several clinically relevant ways. In particular, the <30 group shows higher prior inpatient and emergency utilisation, greater treatment or diagnosis complexity, and different discharge pathway patterns. These descriptive differences do not prove that early readmission can be predicted accurately, but they provide empirical motivation for treating <30 readmission as the positive high-risk class in the subsequent modelling task.
