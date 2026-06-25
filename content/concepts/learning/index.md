---
title: Learning Methods
tags:
  - learning
  - machine-learning
---

# Learning Methods

Learning method notes describe training objectives and representation-learning strategies, independent of a single architecture.

Most learning methods can be described by what target signal $t(x)$ they construct and what loss they optimize:

$$
\min_\theta
\mathbb{E}_{x\sim p_{\mathrm{data}}}
\left[\mathcal{L}(f_\theta(x), t(x))\right]
$$

For supervised learning, $t(x)$ is a human or assay label. For self-supervised learning, $t(x)$ is derived from the input itself.

## Supervision and Transfer

- [[concepts/learning/supervised-learning|Supervised learning]]
- [[concepts/learning/transfer-learning|Transfer learning]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/preference-optimization|Preference optimization]]

## Self-Supervised Learning

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]

## Related

- [[concepts/tasks/index|Tasks]]
- [[concepts/architectures/index|Architectures]]
- [[concepts/evaluation/index|Evaluation]]
- [[research/protein-modeling/index|Protein modeling]]
- [[research/structure-based-ai/index|Structure-based AI]]
