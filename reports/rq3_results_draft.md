# RQ3 Results Draft

## Analysis Goal

RQ3 investigates whether the selected early-readmission model shows consistent performance across clinically relevant subgroups. The analysis emphasises false negatives, since these correspond to encounters that were followed by readmission within 30 days but were not identified by the model as high risk. In a clinical follow-up setting, these missed cases are important because they represent patients who might not be prioritised for additional discharge planning, monitoring, or post-discharge support despite later returning within 30 days.

This section should be interpreted as a post-hoc subgroup error analysis. It does not constitute an additional model-selection experiment, nor should it be presented as a formal fairness audit. The subgroup differences reported here are descriptive rather than causal: they show where the selected model made more or fewer errors, but they do not imply that subgroup membership itself caused those errors.

## Model and Evaluation Setup

The analysed model was the Random Forest classifier trained with Feature Set B from RQ2. This feature set combines the general baseline predictors with the selected glycaemic-management variables. A patient-aware split was used, ensuring that records from the same patient did not appear in both the training and test partitions.

Predicted probabilities were converted to binary predictions using a fixed decision threshold of 0.5, consistent with the default prediction rule used in the RQ2 test-set comparison. Under this threshold, the selected model achieved the following overall performance:

| Metric | Value |
|---|---:|
| Test encounters | 20,024 |
| Positive `<30` encounters | 2,272 |
| PR-AUC | 0.200 |
| ROC-AUC | 0.661 |
| Recall for `<30` | 0.508 |
| Precision for `<30` | 0.187 |
| F1 for `<30` | 0.274 |
| False-negative rate | 0.492 |

In absolute terms, the model correctly flagged 1,154 of the 2,272 early-readmission encounters and missed 1,118.

## Subgroup Definitions

The subgroup analysis considered three clinically relevant dimensions:

- Age group: `<50`, `50-60`, `60-70`, `70-80`, `80+`.
- Race group: `Caucasian`, `AfricanAmerican`, and `Other/Unknown`. Smaller race categories and missing race values were combined to reduce instability from sparse cells.
- Admission source: `Emergency Room`, `Referral`, `Transfer`, and `Other/Unknown`.

All reported subgroups contained at least 50 positive `<30` cases in the test set.

## Main Subgroup Findings

Performance across age groups was comparatively stable. Recall ranged from 0.488 in the `60-70` group to 0.523 in the `80+` group. The corresponding false-negative rates ranged from 0.477 for `80+` encounters to 0.512 for `60-70` encounters. Although these results indicate some age-related variation, the magnitude of the spread was modest relative to the other subgroup dimensions.

Race-group results showed more visible differences. African American encounters had the highest recall, 0.553, and the lowest false-negative rate, 0.447. By contrast, the combined Other/Unknown race group had the lowest recall, 0.458, and the highest false-negative rate, 0.542. This comparison should be interpreted cautiously because the Other/Unknown group aggregates smaller race categories as well as missing or unknown race values.

Admission source also showed meaningful variation. Encounters admitted through the Emergency Room had the highest recall, 0.549, and the lowest false-negative rate, 0.451. Referral encounters had recall 0.448 and false-negative rate 0.552, while the Other/Unknown admission-source group had recall 0.440 and false-negative rate 0.560. These results suggest that the model was less reliable at detecting true early readmissions among non-emergency or less clearly specified admission-source categories. However, the Other/Unknown admission-source result should be interpreted cautiously because this category is a heterogeneous coding group rather than a single coherent clinical pathway.

The false-negative profile provided similar evidence of subgroup imbalance. Referral encounters represented 29.8% of actual positive cases but 33.4% of false negatives. Other/Unknown admission-source encounters accounted for 7.0% of positives and 8.0% of false negatives. For race, the Other/Unknown group represented 5.8% of positives but 6.4% of false negatives.

## Interpretation

The selected model did not distribute errors uniformly across all examined subgroups. Differences were most apparent for admission source and race group, whereas age-group differences were comparatively smaller. In particular, true early-readmission cases were more often missed among Referral and Other/Unknown admission-source encounters and within the combined Other/Unknown race group.

These findings should not be read as definitive evidence that the model is unfair or clinically unsafe for a particular subgroup. They do, however, show that aggregate performance can conceal subgroup-specific weaknesses. The overall recall of 0.508 therefore gives an incomplete picture, since lower recall was observed for Referral and Other/Unknown admission-source encounters and for the combined Other/Unknown race group.

This pattern is consistent with the broader results of the project. RQ1 found that admission and discharge pathways were associated with readmission patterns, while RQ2 showed that general patient complexity and utilisation features carried most of the predictive signal. RQ3 adds a further qualification: even when aggregate model performance is moderate, errors may be concentrated more strongly in particular clinically relevant subgroups.

One important limitation is that this analysis used a fixed 0.5 decision threshold. Different threshold choices may change subgroup false-negative rates, so future work could examine threshold tuning or probability calibration by subgroup before drawing operational conclusions.

## Suggested RQ3 Conclusion

The RQ3 subgroup analysis suggests that the selected Random Forest model has uneven false-negative patterns across clinically relevant subgroups. Age-related differences were present but relatively modest, while admission-source and race-group categories showed larger variation. The model missed more than half of true early-readmission cases among Referral and Other/Unknown admission-source encounters and in the combined Other/Unknown race group. These findings are descriptive rather than causal, and the Other/Unknown admission-source result should be treated as a signal for further checking rather than a standalone clinical explanation. Aggregate model performance should therefore be accompanied by subgroup error analysis, particularly if the model were used to support follow-up decisions for high-risk encounters.

## Generated Outputs

- `src/rq3_subgroup_analysis.py`
- `results/tables/rq3_selected_model_overall_metrics.csv`
- `results/tables/rq3_subgroup_metrics.csv`
- `results/tables/rq3_false_negative_profile.csv`
- `results/tables/rq3_subgroup_metric_spread.csv`
- `reports/figures/rq3_figure_01_subgroup_false_negative_rates.png`

The RQ3 tables and figure can be regenerated with:

```bash
python3 src/rq3_subgroup_analysis.py
```
