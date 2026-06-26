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

## Benchmark Contract

A benchmark can be specified as:

$$
B
=
(\mathcal{D},
\mathcal{T},
\mathcal{S},
\mathcal{M},
\mathcal{A},
\mathcal{R})
$$

where $\mathcal{D}$ is data, $\mathcal{T}$ is the task, $\mathcal{S}$ is the split, $\mathcal{M}$ is metrics, $\mathcal{A}$ is allowed training resources/data, and $\mathcal{R}$ is the reporting protocol.

Without $\mathcal{A}$ and $\mathcal{R}$, benchmark comparisons can mix incompatible evidence, compute budgets, preprocessing, or model-selection procedures.

## Claim Boundary

A benchmark result supports only the generalization claim tested by the split:

$$
\text{benchmark score}
\Rightarrow
\text{claim under }(\mathcal{S},\mathcal{M},\mathcal{A})
$$

Random splits, scaffold splits, protein-family splits, temporal splits, and source-held-out splits answer different questions.

## Failure Modes

- Leaderboard optimization overfits the public test set.
- Test examples leak into pretraining, retrieval corpora, templates, or few-shot prompts.
- Metric rewards easy shortcuts rather than the intended behavior.
- Baselines are too weak to expose dataset artifacts.
- Aggregate score hides subgroup failures or invalid outputs.

## Checks

- What exact generalization claim does the benchmark test?
- Is the split random, temporal, scaffold-based, family-based, source-based, or user-defined?
- Are baselines strong enough to expose trivial shortcuts?
- Is hyperparameter tuning separated from final test evaluation?
- Is the test set protected from contamination?
- Is the benchmark saturated or still discriminative?
- Does the leaderboard reward the behavior that matters?
- Are invalid predictions included in the score denominator?
- Are confidence intervals or paired comparisons reported?
- Are subgroup metrics reported for known hard cases?

## Related

- [[concepts/tasks/index|Tasks]]
- [[concepts/evaluation/evaluation-set-design|Evaluation set design]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/test-set-contamination|Test-set contamination]]
- [[concepts/evaluation/benchmark-saturation|Benchmark saturation]]
