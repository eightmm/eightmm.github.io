---
title: Dataset Card
tags:
  - data
  - documentation
  - evaluation
---

# Dataset Card

A dataset card is a structured summary of what a dataset contains, how it was built, what it can support, and where it can mislead.

A compact view is:

$$
C_{\mathcal{D}}
=
(\text{source}, \text{schema}, \text{examples}, \text{labels}, \text{splits}, \text{limits})
$$

## Suggested Sections

- Purpose: what the dataset is for.
- Source: public origin, license, and collection process.
- Example unit: what one example represents.
- Schema: fields, identifiers, units, and relationships.
- Labels: target meaning, annotation process, uncertainty, and missing values.
- Preprocessing: filtering, normalization, tokenization, featurization, and version.
- Splits: split unit, policy, sizes, and leakage checks.
- Evaluation: supported tasks and metrics.
- Limitations: sampling bias, coverage gaps, label noise, and domain shift.

## Checks

- Can the dataset be reconstructed from public information?
- Are label semantics and units clear?
- Does the split support the intended benchmark claim?
- Are limitations explicit enough to narrow claims?
- Does the card avoid private sources, paths, or unpublished results?

## Related

- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/data/benchmark|Benchmark]]
- [[papers/benchmark-card|Benchmark card]]
