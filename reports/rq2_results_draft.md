# RQ2 Results Draft

## Modelling Task Definition

RQ2 examines whether glycaemic-management features contribute additional predictive information for early readmission after accounting for general patient complexity and prior healthcare utilisation. The outcome variable for this modelling task was `early_readmission`, where encounters followed by readmission within 30 days were coded as the positive class.

Following the exclusion of expired discharges, early readmissions represented approximately 11.3% of encounters. This class imbalance makes accuracy an inappropriate primary criterion for model comparison. The results therefore focus on PR-AUC, recall, precision, and F1, with PR-AUC treated as the main summary measure because it reflects performance on the minority positive class.

The analysis was conducted at the encounter level. The patient identifier `patient_nbr` was retained only for constructing patient-aware data splits and was not included as a model feature. This design prevents records from the same patient from appearing in both the training and test sets.

## Feature-Set Comparison

The modelling experiment compared two feature sets in order to estimate the incremental value of glycaemic-management information.

Feature set A served as the baseline. It included demographic characteristics, admission and discharge variables, encounter-complexity measures, diagnosis-complexity measures, and prior-utilisation indicators. These features included age, gender, race, admission type, admission source, discharge disposition, time in hospital, number of medications, number of diagnoses, prior outpatient visits, prior emergency visits, prior inpatient visits, primary diagnosis group, and whether the encounter involved a diabetes diagnosis.

Feature set B extended feature set A by adding glycaemic-management variables: `A1Cresult`, `max_glu_serum`, `change`, `diabetesMed`, `insulin`, `active_diabetes_med_count`, and `changed_diabetes_med_count`.

Both feature sets were evaluated using the same patient-aware train/test split. The main split contained 80,090 training encounters and 20,024 test encounters, corresponding to 56,417 training patients and 14,022 test patients. No patient appeared in both partitions. The positive class rate was also comparable across the split, at 11.34% in the training set and 11.35% in the test set.

Six models were evaluated: Logistic Regression, Linear SVM, Naive Bayes, Decision Tree, Random Forest, and MLP. Where supported by the estimator, class weights were used to account for the imbalance in early readmission.

## Main Predictive Results

On the main test set, the strongest practical result was obtained by the Random Forest model family. Random Forest with feature set A achieved PR-AUC 0.202, recall 0.526, precision 0.185, F1 0.274, and ROC-AUC 0.659. After adding the glycaemic-management variables in feature set B, Random Forest achieved PR-AUC 0.200, recall 0.508, precision 0.187, F1 0.274, and ROC-AUC 0.661. The PR-AUC difference was therefore very small and slightly negative: -0.001.

The linear models showed only marginal improvement. Logistic Regression increased from PR-AUC 0.1939 to 0.1942, while Linear SVM increased from 0.1922 to 0.1931. Although these changes are positive in direction, their magnitude is not sufficient to support a strong claim that the added glycaemic-management variables materially improve prediction.

The MLP produced a small increase in PR-AUC, from 0.193 to 0.195. However, its recall remained extremely low under both feature sets, indicating that it identified very few positive early-readmission cases. For this reason, the MLP result should be interpreted cautiously and should not be treated as the main evidence for practical early-readmission detection. The more consistent results from Logistic Regression, Linear SVM, and Random Forest suggest that glycaemic-management variables add little predictive value once broader complexity and utilisation measures are already included.

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

The main feature sets included `discharge_disposition_group`. This variable is clinically meaningful, but it may only become available near the end of the hospital encounter. If the intended use case involves prediction earlier during admission, including discharge disposition may overstate the information available at the time of prediction.

To assess whether the RQ2 conclusion depended on this potentially late-available feature, the feature-set comparison was repeated after removing `discharge_disposition_group` from both feature set A and feature set B.

The sensitivity analysis used the same patient-aware splitting principle and again produced no patient overlap between training and test data. The resulting split contained 80,090 training encounters and 20,024 test encounters. The positive class rate remained closely balanced across partitions, at 11.34% in training and 11.35% in testing.

Removing discharge disposition lowered overall predictive performance, indicating that discharge pathway information carries predictive signal. However, the added glycaemic-management variables still produced only small and inconsistent changes. Logistic Regression PR-AUC increased from 0.186 to 0.187, and Linear SVM increased from 0.185 to 0.186. In contrast, Random Forest decreased from 0.192 to 0.190 after the glycaemic-management variables were added.

The no-discharge sensitivity test-set PR-AUC comparison was:

| Model | Feature set A PR-AUC | Feature set B PR-AUC | Delta PR-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.186 | 0.187 | +0.001 |
| Linear SVM | 0.185 | 0.186 | +0.001 |
| Decision Tree | 0.168 | 0.161 | -0.007 |
| Random Forest | 0.192 | 0.190 | -0.003 |
| Naive Bayes | 0.169 | 0.168 | -0.001 |
| MLP | 0.166 | 0.175 | +0.009 |

This sensitivity analysis leads to the same substantive interpretation as the main experiment: glycaemic-management features do not provide a robust or substantial improvement beyond general patient complexity and prior-utilisation features.

## Interpretation

Taken together, the RQ2 results indicate that the recorded glycaemic-management variables have limited incremental predictive value for early readmission in this dataset. This should not be interpreted as evidence that glycaemic management is clinically unimportant. Rather, in the present modelling framework, variables such as HbA1c result availability, glucose serum result availability, medication change, diabetes medication use, and insulin category do not substantially improve predictive performance after broader indicators of patient complexity and healthcare utilisation have been included.

This interpretation is consistent with the descriptive findings from RQ1. The RQ1 analysis showed that early-readmission encounters differed most clearly in prior inpatient and emergency utilisation, encounter complexity, and discharge pathway patterns. RQ2 extends that observation by showing that, once such broader predictors are represented in the model, the additional glycaemic-management variables contribute only marginal gains.

Accordingly, the final report should avoid presenting HbA1c testing or medication-change variables as strong standalone predictors of early readmission. A more defensible interpretation is that these variables may contain some weak directional signal, but the dominant predictive information appears to lie in general complexity and utilisation history.

## Suggested RQ2 Conclusion

The RQ2 modelling results provide limited evidence that glycaemic-management features improve early-readmission prediction beyond general patient complexity and prior healthcare-utilisation features. In the main experiment, Random Forest achieved the strongest practical test-set performance as a model family, but its PR-AUC decreased slightly from 0.202 to 0.200 after the glycaemic-management variables were added. Logistic Regression and Linear SVM showed only near-zero gains. The sensitivity analysis excluding discharge disposition produced the same overall pattern: improvements were close to zero, and Random Forest performance decreased slightly after the glycaemic-management variables were added. Overall, early readmission in this dataset appears to be more strongly associated with broad patient complexity and healthcare-utilisation history than with the recorded glycaemic-management variables alone.

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
