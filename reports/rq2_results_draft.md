# RQ2 Results Draft

## Modelling Task Definition

RQ2 evaluates whether glycaemic-management variables provide additional predictive information for early readmission after accounting for broader indicators of patient complexity and prior healthcare utilisation. The modelling outcome was `early_readmission`, with encounters followed by readmission within 30 days coded as the positive class.

After excluding expired discharges, early readmissions comprised approximately 11.3% of encounters. Given this imbalance, accuracy was not used as the primary criterion for model comparison. The analysis instead emphasised PR-AUC, recall, precision, and F1, with PR-AUC used as the main summary measure because it is more sensitive to performance on the minority positive class.

All analyses were conducted at the encounter level. The patient identifier `patient_nbr` was used only to construct patient-aware splits and was excluded from the feature matrix, preventing records from the same patient from appearing in both training and test partitions.

## Feature-Set Comparison

Two feature sets were compared to assess the incremental contribution of glycaemic-management information.

Feature set A served as the baseline and contained demographic variables, admission and discharge information, encounter-complexity measures, diagnosis-complexity measures, and prior-utilisation indicators. These included age, gender, race, admission type, admission source, discharge disposition, time in hospital, number of medications, number of diagnoses, prior outpatient visits, prior emergency visits, prior inpatient visits, primary diagnosis group, and whether the encounter involved a diabetes diagnosis.

Feature set B retained all variables in feature set A and added seven glycaemic-management variables: `A1Cresult`, `max_glu_serum`, `change`, `diabetesMed`, `insulin`, `active_diabetes_med_count`, and `changed_diabetes_med_count`.

Both feature sets were evaluated on the same patient-aware train/test split. The main split contained 80,090 training encounters and 20,024 test encounters, representing 56,417 training patients and 14,022 test patients. There was no patient overlap between partitions. The positive class rate was also closely matched, at 11.34% in training and 11.35% in testing.

Six classifiers were evaluated: Logistic Regression, Linear SVM, Naive Bayes, Decision Tree, Random Forest, and MLP. Class weights were used where the estimator supported them, in order to reduce the effect of the imbalanced class distribution.

## Main Predictive Results

On the main test set, the Random Forest model family produced the strongest practical performance. Using feature set A, Random Forest achieved PR-AUC 0.202, recall 0.526, precision 0.185, F1 0.274, and ROC-AUC 0.659. With the glycaemic-management variables added in feature set B, the same model achieved PR-AUC 0.200, recall 0.508, precision 0.187, F1 0.274, and ROC-AUC 0.661. The resulting PR-AUC change was therefore very small and slightly negative (-0.001).

The linear models showed only negligible gains. Logistic Regression increased from PR-AUC 0.1939 to 0.1942, and Linear SVM increased from 0.1922 to 0.1931. Although both changes were positive, their magnitude was too small to support the claim that the added glycaemic-management variables materially improved prediction.

The MLP also showed a small PR-AUC increase, from 0.193 to 0.195. However, recall remained extremely low under both feature sets, indicating that the model identified very few true early-readmission cases. For this reason, the MLP result should be interpreted cautiously and should not be treated as evidence of practically useful detection. Across the more stable models, the results indicate that glycaemic-management variables add little predictive value once broader complexity and utilisation measures have already been included.

The main test-set PR-AUC comparison was:

| Model | Feature set A PR-AUC | Feature set B PR-AUC | Delta PR-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.194 | 0.194 | +0.000 |
| Linear SVM | 0.192 | 0.193 | +0.001 |
| Decision Tree | 0.181 | 0.173 | -0.008 |
| Random Forest | 0.202 | 0.200 | -0.001 |
| Naive Bayes | 0.171 | 0.172 | +0.002 |
| MLP | 0.193 | 0.195 | +0.002 |

## Sensitivity Analysis

The main feature sets included `discharge_disposition_group`. Although clinically meaningful, this variable may only be available near the end of the hospital encounter. If prediction is intended earlier during admission, including discharge disposition may overstate the information available at prediction time.

To examine whether this potentially late-available feature affected the RQ2 conclusion, the full feature-set comparison was repeated after removing `discharge_disposition_group` from both feature set A and feature set B.

The sensitivity analysis used the same patient-aware splitting strategy and again produced no patient overlap between training and test data. The split contained 80,090 training encounters and 20,024 test encounters, with positive class rates of 11.34% and 11.35%, respectively.

Removing discharge disposition lowered overall predictive performance, suggesting that discharge pathway information carries useful signal. Even so, the glycaemic-management variables produced only small and inconsistent changes. Logistic Regression PR-AUC increased from 0.186 to 0.187, and Linear SVM increased from 0.185 to 0.186. In contrast, Random Forest decreased from 0.192 to 0.190 after the glycaemic-management variables were added.

The no-discharge sensitivity test-set PR-AUC comparison was:

| Model | Feature set A PR-AUC | Feature set B PR-AUC | Delta PR-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.186 | 0.187 | +0.001 |
| Linear SVM | 0.185 | 0.186 | +0.001 |
| Decision Tree | 0.168 | 0.161 | -0.007 |
| Random Forest | 0.192 | 0.190 | -0.003 |
| Naive Bayes | 0.169 | 0.168 | -0.001 |
| MLP | 0.166 | 0.175 | +0.009 |

The sensitivity analysis therefore supports the same substantive interpretation as the main experiment: glycaemic-management features do not provide a robust or substantial improvement beyond general patient complexity and prior-utilisation features.

## Interpretation

Taken together, the RQ2 results indicate that the recorded glycaemic-management variables have limited incremental predictive value for early readmission in this dataset. This should not be interpreted as evidence that glycaemic management lacks clinical relevance. Rather, within this modelling framework, variables such as HbA1c result availability, glucose serum result availability, medication change, diabetes medication use, and insulin category do not substantially improve predictive performance once broader indicators of patient complexity and healthcare utilisation are represented.

This interpretation is consistent with the descriptive findings from RQ1. Early-readmission encounters differed most clearly in prior inpatient and emergency utilisation, encounter complexity, and discharge pathway patterns. RQ2 extends that observation by showing that, after these broader predictors are included, the additional glycaemic-management variables contribute only marginal gains.

Accordingly, the final report should avoid presenting HbA1c testing or medication-change variables as strong standalone predictors of early readmission. A more defensible interpretation is that these variables may contain weak directional signal, while the dominant predictive information lies in general complexity and utilisation history.

## Suggested RQ2 Conclusion

The RQ2 modelling results provide limited evidence that glycaemic-management features improve early-readmission prediction beyond general patient complexity and prior healthcare-utilisation features. In the main experiment, Random Forest achieved the strongest practical test-set performance as a model family, but its PR-AUC decreased slightly from 0.202 to 0.200 after the glycaemic-management variables were added. Logistic Regression and Linear SVM showed only near-zero gains. The sensitivity analysis excluding discharge disposition produced the same overall pattern: improvements were close to zero, and Random Forest performance again decreased slightly after the glycaemic-management variables were added. Overall, early readmission in this dataset appears to be more strongly associated with broad patient complexity and healthcare-utilisation history than with the recorded glycaemic-management variables alone.

## Generated Outputs

- `src/rq2_baseline.py`
- `src/rq2_visualization.py`
- `src/rq2_sensitivity_no_discharge.py`
- `results/tables/rq2_baseline_cv_results.csv`
- `results/tables/rq2_baseline_test_results.csv`
- `results/tables/rq2_baseline_delta_results.csv`
- `results/tables/rq2_baseline_split_summary.csv`
- `results/tables/rq2_sensitivity_no_discharge_cv_results.csv`
- `results/tables/rq2_sensitivity_no_discharge_test_results.csv`
- `results/tables/rq2_sensitivity_no_discharge_delta_results.csv`
- `results/tables/rq2_sensitivity_no_discharge_split_summary.csv`
- `reports/figures/rq2_figure_01_feature_set_comparison.png`
- `reports/figures/rq2_figure_02_sensitivity_no_discharge.png`

The main RQ2 figure can be regenerated with:

```bash
python3 src/rq2_visualization.py
```

The no-discharge sensitivity tables and figure can be regenerated with:

```bash
python3 src/rq2_sensitivity_no_discharge.py
```
