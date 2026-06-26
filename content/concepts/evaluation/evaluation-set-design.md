---
title: Evaluation Set Design
tags:
  - evaluation
  - benchmark
  - data
---

# Evaluation Set Design

Evaluation set design defines what evidence a model result can support. It chooses the examples, labels, strata, split units, metrics, and failure cases before looking at final performance.

An evaluation set can be treated as:

$$
\mathcal{D}_{\mathrm{eval}}
= \{(x_i, y_i, g_i, s_i, w_i)\}_{i=1}^{m}
$$

$x_i$ is the input, $y_i$ is the target, $g_i$ is the split or grouping unit, $s_i$ is a stratum such as source, class, scaffold, family, or difficulty, and $w_i$ is an optional analysis weight.

## Design Questions

- What claim should the evaluation support: interpolation, OOD generalization, robustness, calibration, ranking quality, or deployment readiness?
- What is one [[concepts/data/example-unit|example unit]]?
- What is the [[concepts/data/split-unit|split unit]] that must not cross train, validation, and test?
- What strata need coverage: class, label range, source, time, modality, scaffold, protein family, task type, or difficulty?
- What metric is primary, and what metrics are diagnostic?
- What failure modes should be counted even when the primary metric looks good?
- What data was unavailable at training, model selection, threshold selection, and final test time?

## Coverage

Coverage should be stated over the intended target population:

$$
\mathrm{coverage}(A)
= \frac{1}{m}\sum_{i=1}^{m}\mathbf{1}[x_i \in A]
$$

$A$ is a subset of interest, such as a rare class, new scaffold, long sequence, low-quality structure, noisy label source, or hard query type. A high average score can hide failure when important subsets have low coverage.

## Minimum Contract

- Population: what distribution the set is meant to represent.
- Inclusion rule: which examples enter the set.
- Exclusion rule: which examples are removed and why.
- Split boundary: what group identity cannot cross splits.
- Label policy: how labels are produced, cleaned, censored, or marked missing.
- Preprocessing boundary: what statistics or filters are fit only on training data.
- Metric policy: primary metric, aggregation, confidence intervals, and diagnostics.
- Contamination audit: duplicate, near-duplicate, metadata, prompt, template, and benchmark-overuse checks.

## Common Mistakes

- Designing only for leaderboard rank instead of the intended claim.
- Using a random split when the claim is scaffold, family, temporal, source, or deployment generalization.
- Letting validation decisions consume the final test set.
- Reporting only aggregate metrics without strata or failure cases.
- Treating examples as independent when they share source, entity, scaffold, family, annotator, user, prompt, or benchmark origin.

## Related

- [[concepts/data/benchmark|Benchmark]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/test-set-contamination|Test-set contamination]]
- [[concepts/evaluation/benchmark-saturation|Benchmark saturation]]
- [[concepts/evaluation/failure-mode-taxonomy|Failure mode taxonomy]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/leakage|Leakage]]
