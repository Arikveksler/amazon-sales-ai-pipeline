# Model Evaluation Report

## Overview
- **Report Generated:** {{TIMESTAMP}}
- **Model Type:** {{MODEL_TYPE}}
- **Model Version:** {{MODEL_VERSION}}
- **Dataset Contract Version:** 2.0.0
- **Pipeline Run ID:** {{RUN_ID}}

---

## Dataset Summary

### Data Split
| Split | Samples | Percentage |
|-------|---------|------------|
| Training | {{TRAIN_SAMPLES}} | {{TRAIN_PCT}}% |
| Validation | {{VAL_SAMPLES}} | {{VAL_PCT}}% |
| Test | {{TEST_SAMPLES}} | {{TEST_PCT}}% |

### Feature Statistics
| Feature | Mean | Std | Min | Max |
|---------|------|-----|-----|-----|
| discounted_price | {{DP_MEAN}} | {{DP_STD}} | {{DP_MIN}} | {{DP_MAX}} |
| actual_price | {{AP_MEAN}} | {{AP_STD}} | {{AP_MIN}} | {{AP_MAX}} |
| rating | {{R_MEAN}} | {{R_STD}} | {{R_MIN}} | {{R_MAX}} |
| rating_count | {{RC_MEAN}} | {{RC_STD}} | {{RC_MIN}} | {{RC_MAX}} |

---

## Model Comparison (Required: at least 2 models)

| Model | {{METRIC_1}} | {{METRIC_2}} | {{METRIC_3}} | Training Time |
|-------|--------------|--------------|--------------|---------------|
| {{MODEL_1}} | {{M1_V1}} | {{M1_V2}} | {{M1_V3}} | {{M1_TIME}} |
| {{MODEL_2}} | {{M2_V1}} | {{M2_V2}} | {{M2_V3}} | {{M2_TIME}} |

### Best Model: {{BEST_MODEL}}
**Selection Criteria:** {{SELECTION_CRITERIA}}

---

## Performance Metrics (Best Model)

### Primary Metrics
| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| {{PRIMARY_METRIC}} | {{TRAIN_PRIMARY}} | {{VAL_PRIMARY}} | {{TEST_PRIMARY}} |
| {{SECONDARY_METRIC}} | {{TRAIN_SECONDARY}} | {{VAL_SECONDARY}} | {{TEST_SECONDARY}} |

### Cross-Validation Results
- **Strategy:** {{CV_FOLDS}}-fold cross-validation
- **Scores:** {{CV_SCORES}}
- **Mean:** {{CV_MEAN}} (+/- {{CV_STD}})

### Feature Importance (Top 5)
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | {{FEAT_1}} | {{IMP_1}} |
| 2 | {{FEAT_2}} | {{IMP_2}} |
| 3 | {{FEAT_3}} | {{IMP_3}} |
| 4 | {{FEAT_4}} | {{IMP_4}} |
| 5 | {{FEAT_5}} | {{IMP_5}} |

---

## Common Errors Analysis

### Error Distribution
{{ERROR_DISTRIBUTION_DESCRIPTION}}

### Typical Failure Cases
1. {{FAILURE_CASE_1}}
2. {{FAILURE_CASE_2}}
3. {{FAILURE_CASE_3}}

### Recommendations for Improvement
{{IMPROVEMENT_RECOMMENDATIONS}}

---

## Data Contract Compliance

### Validation Results
- **Contract Version:** 2.0.0
- **All Constraints Passed:** {{CONSTRAINTS_PASSED}} (Yes/No)
- **Required Columns Present:** {{REQUIRED_COLS_PRESENT}} (Yes/No)

### Constraint Check Details
| Constraint | Column | Status | Details |
|------------|--------|--------|---------|
| Min Value | rating | {{RATING_MIN_STATUS}} | >= 0 |
| Max Value | rating | {{RATING_MAX_STATUS}} | <= 5 |
| Min Value | discounted_price | {{PRICE_MIN_STATUS}} | > 0 |
| Not Nullable | required_columns | {{NULLABLE_STATUS}} | No nulls |

---

## Model Readiness Checklist

- [ ] Model meets accuracy threshold (> {{ACCURACY_THRESHOLD}}%)
- [ ] No significant overfitting (train-test gap < {{OVERFIT_THRESHOLD}}%)
- [ ] Feature importance is interpretable
- [ ] Cross-validation results are stable
- [ ] All contract constraints validated
- [ ] Model card documentation complete

---

## Recommendations

### Next Steps
1. {{RECOMMENDATION_1}}
2. {{RECOMMENDATION_2}}
3. {{RECOMMENDATION_3}}

### Deployment Readiness
**Status:** {{DEPLOYMENT_STATUS}} (Ready/Not Ready/Needs Review)
**Reason:** {{DEPLOYMENT_REASON}}

---

## Appendix

### Hyperparameters (Best Model)
```
{{HYPERPARAMETERS_JSON}}
```

### Reproducibility
- **Random Seed:** {{RANDOM_SEED}}
- **Python Version:** {{PYTHON_VERSION}}
- **scikit-learn Version:** {{SKLEARN_VERSION}}
