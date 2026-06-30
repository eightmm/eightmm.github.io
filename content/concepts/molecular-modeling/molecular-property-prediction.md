---
title: Molecular Property Prediction
tags:
  - molecular-modeling
  - machine-learning
  - property-prediction
---

# Molecular Property Prediction

Molecular property prediction maps a molecule representation to a scalar, class, distribution, or ranked score. It is a core task for ligand modeling, screening, ADMET-style prediction, and molecular representation learning.

A generic supervised setup is:

$$
\hat{y}
=
f_\theta(m)
$$

where $m$ is a molecule represented as a string, graph, fingerprint, descriptor vector, conformer, or ensemble.

The empirical objective is:

$$
\hat{\theta}
=
\arg\min_\theta
\frac{1}{n}
\sum_{i=1}^{n}
\mathcal{L}(f_\theta(m_i), y_i)
$$

## Input Choices

- SMILES or SELFIES-style sequence.
- Molecular graph.
- Fixed fingerprint.
- Physicochemical descriptors.
- 3D conformer or conformer ensemble.
- Protein-conditioned molecule representation for target-specific prediction.

## Task Contract

Molecular property prediction should state what one example means:

$$
e_i = (m_i,\ c_i,\ y_i,\ s_i)
$$

where $m_i$ is the molecular object, $c_i$ is optional context such as target or assay, $y_i$ is the label, and $s_i$ is source/provenance metadata.

| Field | Question |
| --- | --- |
| molecule identity | salt-stripped parent, charged form, tautomer, stereoisomer, or exact submitted structure? |
| representation | SMILES, graph, fingerprint, descriptor, conformer, ensemble, embedding? |
| context | molecule-only, target-conditioned, assay-conditioned, or environment-conditioned? |
| label | class, scalar, censored value, ranking signal, uncertainty target? |
| split unit | molecule, scaffold, assay, target, time, source, or molecule-target pair? |
| metric | classification, regression, ranking, calibration, or applicability-domain metric? |

For target-conditioned activity, the input is not only a molecule:

$$
\hat{y}
=
f_\theta(m,\ t,\ a)
$$

where $t$ is target context and $a$ is assay or measurement context. Ignoring $t$ or $a$ can turn a conditional task into a noisy molecule-only task.

## Label Types

| Label Type | Example | Risk |
| --- | --- | --- |
| physicochemical property | logP, solubility, charge proxy | preprocessing and unit mismatch |
| bioactivity class | active/inactive threshold | threshold and prevalence dependence |
| continuous activity | IC50, Ki, Kd, pIC50 | assay condition and censoring |
| ADMET endpoint | toxicity, permeability, clearance | protocol and species dependence |
| ranking label | relative preference or screen hit order | candidate-pool dependence |
| predicted label | teacher model output | inherited bias and circular evaluation |

The same molecule can have multiple valid labels under different assay conditions. Do not merge them without a label semantics rule.

## Split and Leakage

Random row splits are often too weak. For a model claim $C$, choose the split unit stronger than the intended generalization:

$$
\text{split unit}
\succeq
\text{claim unit}
$$

| Claim | Stronger Split |
| --- | --- |
| new analogs near training chemistry | random or molecule split may be enough |
| new scaffolds | scaffold split |
| new targets | target or protein-family split |
| new assays | assay/source split |
| prospective use | time split |
| protein-ligand generalization | pair split plus ligand and protein grouping |

## Evaluation Risks

- Random splits can overestimate scaffold generalization.
- Assay labels can be noisy, censored, and protocol-dependent.
- Activity cliffs make local similarity unreliable.
- Negative sets may mix true inactivity with untested molecules.
- Standardization choices can change duplicates and labels.

## Checks

- What does the target label mean and how was it measured?
- Is the task classification, regression, ranking, or uncertainty estimation?
- Are scaffold, time, target, or assay splits needed?
- If the label is target-conditioned, is the [[concepts/sbdd/protein-ligand-split|protein-ligand split]] explicit?
- Are stereochemistry, salts, tautomers, and protonation handled consistently?
- Does performance hold outside the training applicability domain?
- Is the molecule identity key defined before deduplication?
- Are censored, repeated, or conflicting labels handled explicitly?
- Does the metric match property prediction, ranking, or decision use?

## Related

- [[concepts/tasks/property-prediction|Property prediction]]
- [[concepts/machine-learning/classification|Classification]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/machine-learning/ranking|Ranking]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/molecular-modeling/molecular-identity|Molecular identity]]
- [[concepts/molecular-modeling/chemical-state-contract|Chemical state contract]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/censored-label|Censored label]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[concepts/evaluation/activity-cliff|Activity cliff]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
