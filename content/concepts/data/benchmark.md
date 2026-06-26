---
title: Benchmark
tags:
  - data
  - benchmark
  - evaluation
---

# Benchmark

A benchmark is a dataset, task, split, metric, and protocol used to compare methods. It is not just a collection of examples.

A benchmark estimate can be written as:

$$
\hat{R}_{\mathrm{bench}}(f)
= \frac{1}{|\mathcal{D}_{\mathrm{test}}|}
\sum_{(x_i,y_i)\in\mathcal{D}_{\mathrm{test}}}
\mathcal{L}_{\mathrm{metric}}(f(x_i),y_i)
$$

This estimate supports a claim only when the test set and metric match the intended use case.

## Required Pieces

- Task definition.
- Dataset version.
- Evaluation set design.
- Split policy.
- Metric and aggregation rule.
- Baselines.
- Allowed training data and preprocessing.
- Reporting protocol for uncertainty and failure cases.
- Contamination and saturation checks.

## Checks

- What exact generalization claim does the benchmark test?
- Is the split random, temporal, scaffold-based, family-based, source-based, or user-defined?
- Are baselines strong enough to expose trivial shortcuts?
- Is hyperparameter tuning separated from final test evaluation?
- Is the test set protected from contamination?
- Is the benchmark saturated or still discriminative?
- Does the leaderboard reward the behavior that matters?

## Related

- [[concepts/tasks/index|Tasks]]
- [[concepts/evaluation/evaluation-set-design|Evaluation set design]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/test-set-contamination|Test-set contamination]]
- [[concepts/evaluation/benchmark-saturation|Benchmark saturation]]
