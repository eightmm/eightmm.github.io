---
title: Pretraining
tags:
  - pretraining
  - representation-learning
  - machine-learning
---

# Pretraining

Pretraining learns reusable parameters or representations before adapting a model to a target task. It is the base pattern behind large language models, vision encoders, protein language models, molecular encoders, and many multimodal systems.

The generic two-stage view is:

$$
\theta_{\mathrm{pre}}
=
\arg\min_\theta
\mathbb{E}_{x\sim p_{\mathrm{source}}}
\left[
\mathcal{L}_{\mathrm{pre}}(f_\theta(x))
\right]
$$

followed by adaptation:

$$
\theta_{\mathrm{task}}
=
\arg\min_\theta
\mathbb{E}_{(x,y)\sim p_{\mathrm{target}}}
\left[
\mathcal{L}_{\mathrm{task}}(f_\theta(x),y)
\right],
\qquad
\theta_0=\theta_{\mathrm{pre}}
$$

## Common Objectives

- Autoregressive next-token prediction.
- Masked modeling over text, sequence, image patches, molecules, or proteins.
- Contrastive alignment between views or modalities.
- Denoising, reconstruction, or representation prediction.
- Multitask supervised pretraining over a broad source dataset.

## Why It Matters

- Reduces labeled data requirements for downstream tasks.
- Makes architecture comparisons depend heavily on source data and objective.
- Can transfer useful structure, but can also transfer bias or shortcuts.
- In scientific AI, pretraining data boundaries can dominate claimed generalization.

## Checks

- What source distribution was used?
- What target signal did pretraining optimize?
- Does the pretraining objective match downstream evaluation?
- Could downstream test examples or close homologs/duplicates appear in pretraining?
- Is adaptation done by [[concepts/learning/linear-probing|probing]], [[concepts/learning/fine-tuning-protocol|full fine-tuning]], or parameter-efficient fine-tuning?

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/learning/transfer-learning|Transfer learning]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/learning/linear-probing|Linear probing]]
- [[concepts/learning/fine-tuning-protocol|Fine-tuning protocol]]
- [[concepts/data/data-distribution|Data distribution]]
