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
- [[concepts/learning/semi-supervised-learning|Semi-supervised learning]]
- [[concepts/learning/pretraining|Pretraining]]
- [[concepts/learning/transfer-learning|Transfer learning]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/knowledge-distillation|Knowledge distillation]]
- [[concepts/learning/instruction-tuning|Instruction tuning]]
- [[concepts/learning/domain-adaptation|Domain adaptation]]
- [[concepts/learning/curriculum-learning|Curriculum learning]]
- [[concepts/learning/imitation-learning|Imitation learning]]
- [[concepts/learning/active-learning|Active learning]]

## Self-Supervised Learning

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/augmentation-policy|Augmentation policy]]
- [[concepts/learning/representation-collapse|Representation collapse]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]

## Reinforcement and Preference Learning

- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[concepts/learning/policy-gradient|Policy gradient]]
- [[concepts/learning/reward-modeling|Reward modeling]]
- [[concepts/learning/preference-optimization|Preference optimization]]

## Related

- [[concepts/tasks/index|Tasks]]
- [[concepts/architectures/index|Architectures]]
- [[concepts/evaluation/index|Evaluation]]
- [[agents/index|Agents]]
- [[research/protein-modeling/index|Protein modeling]]
- [[research/structure-based-ai/index|Structure-based AI]]
