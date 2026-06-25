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

- [[concepts/tasks/index|Tasks]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/evaluation/activity-cliff|Activity cliff]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]

## Related

- [[agents/verification-loop|Verification loop]]
- [[concepts/tasks/index|Tasks]]
- [[concepts/index|Concepts]]
