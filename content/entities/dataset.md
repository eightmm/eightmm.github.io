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
- Is the split key the same entity that defines the intended generalization claim?
- Is label provenance kept with each row?

## Related

- [[entities/entity-relation-map|Entity relation map]]
- [[concepts/data/index|Data]]
- [[concepts/data/data-curation|Data curation]]
- [[concepts/data/annotation-labeling|Annotation and labeling]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/data/benchmark|Benchmark]]
- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/evaluation/activity-cliff|Activity cliff]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[entities/assay|Assay]]
- [[entities/molecule|Molecule]]
- [[entities/sequence|Sequence]]
- [[entities/structure|Structure]]
