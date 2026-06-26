---
title: Negative Set
tags:
  - evaluation
  - molecular-modeling
  - dataset
---

# Negative Set

A negative set defines examples treated as inactive, irrelevant, non-binding, or low-quality. In molecular ML, negative construction is often the benchmark, not a detail.

The key distinction is provenance:

$$
y_i = 0
\quad\not\Rightarrow\quad
\text{unmeasured}
$$

An unmeasured molecule is unknown, not automatically inactive.

## Common Sources

| Source | What It Supports | Main Risk |
| --- | --- | --- |
| measured inactive examples | assay-specific inactive classification | assay threshold, censoring, and replicate handling dominate |
| property-matched decoys | virtual-screening enrichment benchmark | decoys may be separable by artifacts or too easy |
| random library molecules | broad retrieval stress test | unrealistic prevalence and trivial negatives |
| assumed negatives from missing labels | weak supervision only | unknowns are mislabeled as true negatives |
| cross-target negatives | target-conditioned selectivity tasks | hidden activity against the target may be unmeasured |

## Label Semantics

Negative construction should preserve the distinction:

$$
y = 0
\neq
y\ \text{missing}
\neq
y < \tau
\neq
\text{not selected}
$$

where $\tau$ is an assay-specific activity threshold. A benchmark should state whether a negative is measured inactive, censored below a threshold, unobserved, or sampled as a decoy.

## Evaluation Pattern

| Field | Required Detail |
| --- | --- |
| negative provenance | measured, decoy, random, assumed, cross-target, or generated |
| prevalence | natural prevalence or rebalanced benchmark prevalence |
| property matching | size, charge, hydrophobicity, topology, and scaffold controls |
| split unit | scaffold, target, assay/source, temporal, or complex-pair split |
| baseline | simple property, fingerprint, similarity, or docking baseline |
| metric | enrichment, ranking, classification, calibration, or decision utility |

## Checks

- Are negatives experimentally measured or assumed?
- Are decoys separable by trivial properties such as size, charge, or hydrophobicity?
- Is performance reported separately for measured inactives and decoys?
- Does the active threshold match the assay and task?
- Would a simple property-only or fingerprint baseline solve the benchmark?
- Is the evaluation prevalence the same as the deployment prevalence?
- Are missing labels excluded, modeled as unknown, or incorrectly converted to inactive?

## Related

- [[entities/assay|Assay]]
- [[entities/dataset|Dataset]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/precision-recall|Precision and recall]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[concepts/evaluation/leakage|Leakage]]
