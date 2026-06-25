---
title: Evaluation
tags:
  - evaluation
  - methodology
---

# Evaluation

Evaluation notes collect methods for measuring model quality honestly — how to split data, detect leakage, calibrate confidence, and test generalization beyond the training distribution.

The recurring theme: a metric is only as trustworthy as the split and protocol behind it. Most reported gains that fail to reproduce trace back to an evaluation flaw, not a modeling one.

The basic estimate is a held-out risk:

$$
\hat{R}(f)
= \frac{1}{m}\sum_{j=1}^{m}
\mathcal{L}(f(x_j), y_j)
$$

This number is only meaningful if the held-out set matches the intended generalization claim.

## Topics

- [[concepts/data/benchmark|Benchmark]]
- [[concepts/data/sampling-strategy|Sampling strategy]]
- [[concepts/tasks/index|Tasks]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/regression-metrics|Regression metrics]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
- [[concepts/math/statistical-estimator|Statistical estimator]]
- [[concepts/math/bias-variance-tradeoff|Bias-variance tradeoff]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/cross-validation|Cross-validation]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/evaluation/activity-cliff|Activity cliff]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/robustness|Robustness]]
- [[concepts/evaluation/error-analysis|Error analysis]]
- [[concepts/evaluation/interpretability|Interpretability]]

## Related

- [[concepts/math/expectation|Expectation]]
- [[agents/verification/verification-loop|Verification loop]]
- [[concepts/tasks/index|Tasks]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
- [[concepts/index|Concepts]]
