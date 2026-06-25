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

## Related

- [[concepts/tasks/property-prediction|Property prediction]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[concepts/evaluation/activity-cliff|Activity cliff]]
