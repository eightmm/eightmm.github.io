---
title: Property Prediction
tags:
  - tasks
  - prediction
  - machine-learning
---

# Property Prediction

Property prediction maps an entity representation to a scalar, class, distribution, or ranked score. The entity can be a molecule, protein, sequence, structure, material, image, table row, or protein-ligand complex.

The generic task is:

$$
\hat{y}
=
f_\theta(r_e, c)
$$

where $r_e=\phi(e)$ is a representation of entity $e$ and $c$ is optional context such as target, assay, environment, source, or time.

For regression:

$$
\mathcal{L}
=
\frac{1}{n}\sum_{i=1}^{n}
\ell(f_\theta(r_i,c_i), y_i)
$$

For classification:

$$
\hat{p}(y=k\mid r,c)
=
\mathrm{softmax}(W h_\theta(r,c)+b)_k
$$

## Output Variants

- Scalar regression: affinity, energy, toxicity score, expression level, or physical property.
- Classification: active/inactive, functional class, risk class, or annotation class.
- Distributional prediction: mean and uncertainty, censored range, or calibrated probability.
- Ranking score: prioritization value used for retrieval, screening, or triage.

## Bio-AI Notes

Molecular property prediction is a special case where the entity is a [[entities/molecule|Molecule]] and the representation may be SMILES, graph, fingerprint, conformer, or embedding.

Target-conditioned property prediction must preserve the [[entities/target-assay-label|Target-assay-label contract]]:

$$
y = h(m, t, a, c)
$$

where $m$ is molecule, $t$ is target, $a$ is assay, and $c$ is measurement context.

## Checks

- What entity does one example represent?
- Is the property intrinsic, target-conditioned, assay-conditioned, or context-dependent?
- Is the output a scalar, class, probability, censored value, or ranking score?
- What loss is optimized, and what metric is reported?
- Does the split prevent entity, scaffold, family, source, assay, or time leakage?
- Are missing, weak, censored, and noisy labels handled explicitly?

## Related

- [[concepts/tasks/task-specification|Task specification]]
- [[concepts/tasks/task-output-space|Task output space]]
- [[concepts/molecular-modeling/molecular-property-prediction|Molecular property prediction]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/censored-label|Censored label]]
- [[concepts/evaluation/regression-metrics|Regression metrics]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/calibration|Calibration]]
