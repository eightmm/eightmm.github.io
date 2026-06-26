---
title: Target-Assay-Label Contract
tags:
  - entities
  - molecular-modeling
  - labels
  - datasets
---

# Target-Assay-Label Contract

A target-assay-label contract states what biological object, measurement process, and label semantics define one supervised chem-bio example. It prevents a dataset row from being treated as a generic molecule label when it is actually target- and assay-conditioned.

The core relation is:

$$
y_i
=
h(m_i, t_i, a_i, c_i)
$$

where $m_i$ is a [[entities/molecule|Molecule]] or [[entities/ligand|Ligand]], $t_i$ is the [[entities/target|Target]], $a_i$ is the [[entities/assay|Assay]], and $c_i$ is measurement context such as endpoint, unit, threshold, dose, species, construct, time, source, and preprocessing.

## Contract Fields

| Field | Meaning |
| --- | --- |
| Molecule or ligand | Standardized chemical identity and representation |
| Target | Protein, pocket, family, sequence region, or other scoped biological object |
| Assay | Measurement protocol, readout, endpoint, source, and context |
| Label | Value, unit, transformation, censoring, threshold, and missing-label policy |
| Example unit | What one row or complex represents |
| Split unit | Molecule group, protein family, assay/source group, complex pair, or time |
| Evaluation | Task, metric, aggregation, and failure mode |

## Why It Matters

- The same molecule can have different labels for different targets.
- The same molecule-target pair can have incompatible labels across assay protocols.
- A binary active/inactive label can hide endpoint, threshold, unit, and censoring.
- Missing activity is not the same as measured inactivity.
- A split can look clean by row id while leaking molecule, target, assay, or source context.

## Split Implication

A target-conditioned task should not be evaluated only as a molecule split unless the claim is molecule interpolation for known targets:

$$
s_i
=
g(m_i, t_i, a_i, \tau_i)
$$

where $s_i$ is the split assignment and $\tau_i$ is optional time or source context. The grouping keys should match the intended generalization claim.

## Checks

- Is the label molecule-only, target-conditioned, assay-conditioned, or complex-conditioned?
- Are endpoint, unit, transformation, threshold, and censoring documented?
- Can conflicting measurements be explained by target or assay context?
- Does the split prevent leakage through molecule scaffold, protein family, assay batch, source, or time?
- Does the metric match the label semantics rather than only the model output type?
- Are private target names, collaborator context, internal task names, and unpublished results omitted?

## Related

- [[entities/entity-relation-map|Entity relation map]]
- [[entities/molecule|Molecule]]
- [[entities/ligand|Ligand]]
- [[entities/target|Target]]
- [[entities/assay|Assay]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[entities/dataset|Dataset]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/missing-data|Missing data]]
- [[concepts/data/censored-label|Censored label]]
- [[concepts/data/weak-label|Weak label]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
