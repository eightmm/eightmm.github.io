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

## Assay Contract

| Field | Meaning | Risk If Missing |
| --- | --- | --- |
| target | biological object being measured | labels from different targets collapse together |
| endpoint | what is measured | binding, function, toxicity, and viability get mixed |
| readout | experimental signal | proxy signal may not equal desired mechanism |
| unit | concentration, percent, score, count | regression values become incomparable |
| threshold | active/inactive boundary | binary labels become source-specific |
| censoring | greater-than or less-than limits | point regression becomes biased |
| protocol/source | public assay/source identifier | batch or source leakage is hidden |
| context | organism, construct, cell line, time, dose | label meaning changes silently |

An assay-conditioned label can be written as:

$$
y
=
g(m,t,a,c)
$$

where $m$ is the molecule, $t$ is the target, $a$ is the assay, and $c$ is measurement context.

## Assay as Model Input

Some papers use assay metadata as a task condition:

$$
\hat{y}
=
f_\theta(x_{\mathrm{molecule}}, x_{\mathrm{target}}, x_{\mathrm{assay}})
$$

This is different from removing assay effects during harmonization. If assay identity is an input, the test split must prevent the model from memorizing assay-specific shortcuts.

## Split Implications

| Split Type | What It Tests |
| --- | --- |
| random row split | interpolation over known molecules, targets, and assays |
| molecule split | transfer to new molecules under known assay context |
| target split | transfer to new targets or families |
| assay/source split | transfer to new measurement protocols |
| time split | future assay records or database releases |

Assay/source split is often needed when the paper claims assay-robust or source-robust performance.

## Checks

- What target, readout, organism, and protocol produced the label?
- Are labels from different assay types mixed as if they were comparable?
- Is the task predicting assay response, binding, toxicity, or a proxy endpoint?
- Does the train/test split leak through repeated molecules or shared assay batches?
- Are [[concepts/data/censored-label|censored]] or thresholded values handled as such, instead of point labels?
- Is the assay used as metadata, split group, task context, or prediction target?
- Are replicate measurements aggregated, weighted, or kept separate?
- Does the benchmark report performance per assay/source, not only globally?

## Related

- [[entities/entity-relation-map|Entity relation map]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[concepts/data/censored-label|Censored label]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/dataset-shift|Dataset shift]]
- [[entities/molecule|Molecule]]
- [[entities/protein|Protein]]
- [[entities/dataset|Dataset]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
