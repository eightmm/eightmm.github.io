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
- Entity contract: molecule, target, assay, label, unit, threshold, censoring, and source when relevant.
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
- Are preprocessing and split decisions versioned?
- Is the intended evaluation protocol stated, not only the metric?

## Minimum Public Card

For a public wiki note, a minimal dataset card should include:

- Public source and citation status.
- Example unit and split unit.
- Label semantics and missing-label policy.
- Target-assay-label contract for chem-bio datasets when relevant.
- Preprocessing contract summary.
- Split policy and leakage checks.
- Supported tasks and metrics.
- Known limitations and out-of-scope claims.

## Related

- [[concepts/data/dataset-construction-checklist|Dataset construction checklist]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/data/benchmark|Benchmark]]
- [[papers/benchmark-card|Benchmark card]]
