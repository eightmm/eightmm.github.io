---
title: Missing Modality
tags:
  - modalities
  - multimodal
  - robustness
---

# Missing Modality

A missing modality occurs when a model expects multiple input signals but one or more are absent, corrupted, delayed, or unavailable at deployment. This is common in multimodal systems, scientific datasets, clinical-style tables, and agent workflows.

For modalities $\mathcal{M}$, let $r_m \in \{0,1\}$ indicate whether modality $m$ is observed:

$$
x_{\mathrm{obs}}
=
\{x^{(m)} : r_m = 1,\ m\in\mathcal{M}\}
$$

A robust model should define a prediction even when not all $r_m$ are one:

$$
\hat{y}
=
f_\theta(x_{\mathrm{obs}}, r)
$$

where $r$ is the observation mask.

## Missingness Patterns

- MCAR: missing completely at random.
- MAR: missingness depends on observed variables.
- MNAR: missingness depends on unobserved variables or the missing value itself.
- Operational missingness: a sensor, tool, file, measurement, or modality is not available in production.

## Common Strategies

- Modality dropout during training.
- Mask tokens or learned missing-modality embeddings.
- Late fusion models that can skip absent branches.
- Imputation, with uncertainty tracked separately.
- Fallback single-modality models.

## Checks

- Which modalities are guaranteed at training, validation, and deployment?
- Is missingness random or correlated with the target?
- Does the model know which modality is missing?
- Are missing-modality cases evaluated separately?
- Does imputation leak future or target information?

## Related

- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[concepts/modalities/modality-alignment|Modality alignment]]
- [[concepts/evaluation/robustness|Robustness]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
