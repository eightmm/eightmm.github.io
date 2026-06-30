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
- Lineage: how raw records were filtered, transformed, labeled, and split.
- Example unit: what one example represents.
- Schema: fields, identifiers, units, and relationships.
- Labels: target meaning, annotation process, uncertainty, [[concepts/data/missing-data|missing values]], [[concepts/data/censored-label|censoring]], and [[concepts/data/weak-label|weak labels]].
- Entity contract: molecule, target, assay, label, unit, threshold, censoring, and source when relevant.
- Preprocessing: filtering, normalization, tokenization, featurization, and version.
- Splits: split unit, policy, sizes, and leakage checks.
- Evaluation: supported tasks and metrics.
- Limitations: sampling bias, coverage gaps, label noise, and domain shift.

## Claim Boundary

A dataset card should state what claims the dataset can and cannot support:

| Claim type | Card should specify |
| --- | --- |
| interpolation | duplicate, near-neighbor, and source coverage |
| new chemistry | scaffold or cluster split policy |
| new protein/family | sequence identity or family split policy |
| future deployment | time split and source drift |
| probabilistic decision | calibration set and label uncertainty |
| generation quality | validity, novelty, diversity, and filter policy |

The card is not just documentation; it constrains how benchmark results may be interpreted.

## Versioned Identity

Dataset identity should include data and processing versions:

$$
\operatorname{id}(\mathcal{D})
=
H(\text{raw source}, \text{filters}, \text{labels}, \text{splits}, \text{preprocessing})
$$

If any of these change, results should not be treated as directly comparable without an update note.

## Checks

- Can the dataset be reconstructed from public information?
- Are label semantics and units clear?
- Does the split support the intended benchmark claim?
- Are limitations explicit enough to narrow claims?
- Does the card avoid private sources, paths, or unpublished results?
- Are preprocessing and split decisions versioned?
- Is the intended evaluation protocol stated, not only the metric?
- Does the card distinguish raw data, curated data, features, and benchmark splits?
- Does the card state unsupported or out-of-scope claims?

## Minimum Public Card

For a public wiki note, a minimal dataset card should include:

- Public source and citation status.
- Example unit and split unit.
- Label semantics and missing-label policy.
- Censoring and weak-label policy when applicable.
- Target-assay-label contract for chem-bio datasets when relevant.
- Preprocessing contract summary.
- Split policy and leakage checks.
- Supported tasks and metrics.
- Known limitations and out-of-scope claims.

## Related

- [[concepts/data/dataset-construction-checklist|Dataset construction checklist]]
- [[concepts/data/data-versioning|Data versioning]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/data/data-lineage|Data lineage]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/data/benchmark|Benchmark]]
- [[papers/analysis/benchmark-card|Benchmark card]]
