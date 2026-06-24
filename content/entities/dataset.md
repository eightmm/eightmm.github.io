---
title: Dataset
tags:
  - entities
  - dataset
---

# Dataset

A dataset is a curated collection of examples, labels, metadata, and splits used to train or evaluate models.

## Why It Matters

- Splits, duplicate handling, and label provenance shape what a benchmark measures.
- Molecular datasets often mix [[entities/molecule|molecules]], [[entities/protein|proteins]], assays, and structures.
- Clear dataset notes make leakage risks and task definitions easier to audit.

## Checks

- What is one training example, and what is the target label?
- Are duplicate molecules, protein families, or near-identical structures separated?
- Does the split match the intended generalization setting?
- Are missing labels, censored measurements, and assay metadata handled explicitly?

## Related

- [[entities/assay|Assay]]
- [[entities/molecule|Molecule]]
- [[entities/sequence|Sequence]]
- [[entities/structure|Structure]]
