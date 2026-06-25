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

- Measured inactive examples from an assay.
- Decoys designed to match simple molecular properties.
- Random library molecules.
- Assumed negatives from missing labels.

## Checks

- Are negatives experimentally measured or assumed?
- Are decoys separable by trivial properties such as size, charge, or hydrophobicity?
- Is performance reported separately for measured inactives and decoys?
- Does the active threshold match the assay and task?
- Would a simple property-only or fingerprint baseline solve the benchmark?

## Related

- [[entities/assay|Assay]]
- [[entities/dataset|Dataset]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[concepts/evaluation/leakage|Leakage]]
