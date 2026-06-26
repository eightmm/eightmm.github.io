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
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/evaluation-set-design|Evaluation set design]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/failure-mode-taxonomy|Failure mode taxonomy]]
- [[concepts/evaluation/confusion-matrix|Confusion matrix]]
- [[concepts/evaluation/threshold-selection|Threshold selection]]
- [[concepts/evaluation/precision-recall|Precision and recall]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/probability-metrics|Probability metrics]]
- [[concepts/evaluation/proper-scoring-rule|Proper scoring rule]]
- [[concepts/evaluation/brier-score|Brier score]]
- [[concepts/evaluation/regression-metrics|Regression metrics]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
- [[concepts/math/statistical-estimator|Statistical estimator]]
- [[concepts/math/central-limit-theorem|Central limit theorem]]
- [[concepts/math/hypothesis-testing|Hypothesis testing]]
- [[concepts/math/bias-variance-tradeoff|Bias-variance tradeoff]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/bootstrap-evaluation|Bootstrap evaluation]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[concepts/evaluation/effect-size|Effect size]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/multiple-comparisons|Multiple comparisons]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/cross-validation|Cross-validation]]
- [[concepts/machine-learning/model-selection|Model selection]]
- [[concepts/machine-learning/hyperparameter-tuning|Hyperparameter tuning]]
- [[concepts/machine-learning/early-stopping|Early stopping]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/test-set-contamination|Test-set contamination]]
- [[concepts/evaluation/benchmark-saturation|Benchmark saturation]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/reliability-diagram|Reliability diagram]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/evaluation/activity-cliff|Activity cliff]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/boltzmann-ceiling|Boltzmann ceiling analysis]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/conformal-prediction|Conformal prediction]]
- [[concepts/evaluation/selective-prediction|Selective prediction]]
- [[concepts/evaluation/robustness|Robustness]]
- [[concepts/evaluation/error-analysis|Error analysis]]
- [[concepts/evaluation/interpretability|Interpretability]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/learning/linear-probing|Linear probing]]
- [[concepts/learning/fine-tuning-protocol|Fine-tuning protocol]]

## Related

- [[concepts/math/expectation|Expectation]]
- [[agents/verification/verification-loop|Verification loop]]
- [[concepts/tasks/index|Tasks]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
- [[concepts/index|Concepts]]
