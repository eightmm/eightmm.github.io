---
title: Bioactivity Label
tags:
  - entities
  - molecular-modeling
  - labels
---

# Bioactivity Label

A bioactivity label is a measured or curated value that describes how a molecule, ligand, protein, target, assay, or context behaves in a biological or chemical setup.

It is not just a number. A label should be read as:

$$
y
=
\operatorname{measure}(m, t, a, c)
$$

where $m$ is a molecule or ligand, $t$ is a target, $a$ is an assay protocol, and $c$ is measurement context such as endpoint, organism, construct, dose, time, unit, and preprocessing.

This is the label side of the [[entities/target-assay-label|Target-assay-label contract]].

## Common Forms

- Continuous: $K_d$, $K_i$, $\mathrm{IC}_{50}$, $\mathrm{EC}_{50}$, percent inhibition, activity score, affinity proxy.
- Transformed: $pK_d$, $pK_i$, $pIC_{50}$, log concentration, standardized score.
- Binary: active/inactive after a threshold.
- [[concepts/data/censored-label|Censored]]: values reported as greater-than or less-than limits.
- [[concepts/data/weak-label|Weak]] or [[concepts/data/missing-data|missing]]: untested, inferred, aggregated, or source-dependent labels.

## Why It Matters

- The same molecule-target pair can have different labels under different assay contexts.
- A binary active label may hide endpoint type, threshold, unit, and censoring.
- Regression, classification, ranking, and virtual screening can use the same raw records differently.
- Negative examples may mean measured inactive, not selected, or simply unobserved.

## Checks

- What endpoint does the label represent?
- What unit, transformation, threshold, and censoring policy are used?
- Is the label molecule-only, target-conditioned, assay-conditioned, or complex-conditioned?
- Are replicate measurements aggregated, kept separate, or filtered?
- Are conflicting labels resolved globally or kept assay-specific?
- Does the split prevent molecule, target, assay, or source leakage?
- Is the label valid for the intended task, or only for the original assay context?

## Related

- [[entities/assay|Assay]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[entities/molecule|Molecule]]
- [[entities/ligand|Ligand]]
- [[entities/target|Target]]
- [[entities/dataset|Dataset]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/missing-data|Missing data]]
- [[concepts/data/censored-label|Censored label]]
- [[concepts/data/weak-label|Weak label]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/molecular-modeling/molecular-property-prediction|Molecular property prediction]]
