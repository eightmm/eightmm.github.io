---
title: Assay
tags:
  - entities
  - assay
  - measurement
---

# Assay

An assay is an experimental measurement that links a biological or chemical setup to an observed label.

In supervised chem-bio ML, an assay is part of the [[entities/target-assay-label|Target-assay-label contract]]:

$$
\text{assay}
\rightarrow
(\text{endpoint}, \text{unit}, \text{threshold}, \text{censoring}, \text{context})
$$

## Why It Matters

- Assay conditions define what a molecular activity label actually means.
- Similar molecules can have labels from incompatible protocols or target contexts.
- Dataset construction should preserve assay identity when possible.
- The same molecule-target pair can have conflicting labels across protocols or sources.

## Checks

- What target, readout, organism, and protocol produced the label?
- Are labels from different assay types mixed as if they were comparable?
- Is the task predicting assay response, binding, toxicity, or a proxy endpoint?
- Does the train/test split leak through repeated molecules or shared assay batches?
- Are censored or thresholded values handled as such, instead of point labels?
- Is the assay used as metadata, split group, task context, or prediction target?

## Related

- [[entities/entity-relation-map|Entity relation map]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[entities/molecule|Molecule]]
- [[entities/protein|Protein]]
- [[entities/dataset|Dataset]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
