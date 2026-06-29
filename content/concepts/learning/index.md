---
title: Learning Methods
tags:
  - learning
  - machine-learning
---

# Learning Methods

Learning method note는 특정 architecture와 독립적인 training objective와 representation-learning strategy를 설명합니다.

대부분의 learning method는 어떤 target signal $t(x)$를 만들고 어떤 loss를 최적화하는지로 설명할 수 있습니다.

$$
\min_\theta
\mathbb{E}_{x\sim p_{\mathrm{data}}}
\left[\mathcal{L}(f_\theta(x), t(x))\right]
$$

For supervised learning, $t(x)$ is a human or assay label. For self-supervised learning, $t(x)$ is derived from the input itself.

## Route Map

| Question | Start | Evidence Boundary |
| --- | --- | --- |
| is the target externally labeled? | [Supervised learning](/concepts/learning/supervised-learning), [Semi-supervised learning](/concepts/learning/semi-supervised-learning) | label semantics, noise, split unit |
| is the target derived from the input? | [Self-supervised learning](/concepts/learning/self-supervised-learning) | augmentation, masking, collapse, transfer metric |
| is the model reused on a downstream task? | [Pretraining](/concepts/learning/pretraining), [Transfer learning](/concepts/learning/transfer-learning), [Fine-tuning](/concepts/learning/fine-tuning) | downstream protocol and data split |
| is representation quality the claim? | [Linear probing](/concepts/learning/linear-probing), [Representation evaluation](/concepts/learning/representation-evaluation) | frozen encoder, probe capacity, task diversity |
| is behavior learned from actions or preferences? | [Reinforcement learning](/concepts/learning/reinforcement-learning), [Preference optimization](/concepts/learning/preference-optimization) | reward source, off-policy data, evaluation loop |
| is the data distribution changing? | [Domain adaptation](/concepts/learning/domain-adaptation), [Active learning](/concepts/learning/active-learning) | source/target split and sampling policy |

## Supervision and Transfer

| Method | Use For |
| --- | --- |
| [Supervised learning](/concepts/learning/supervised-learning) | direct labels from humans, assays, annotations, or measurements |
| [Semi-supervised learning](/concepts/learning/semi-supervised-learning) | small labeled set plus larger unlabeled set |
| [Pretraining](/concepts/learning/pretraining) | learning reusable parameters before downstream adaptation |
| [Transfer learning](/concepts/learning/transfer-learning) | moving representations or weights to a target task |
| [Fine-tuning](/concepts/learning/fine-tuning), [Fine-tuning protocol](/concepts/learning/fine-tuning-protocol) | adapting pretrained models with explicit data and metric contracts |
| [Knowledge distillation](/concepts/learning/knowledge-distillation) | training a smaller or deployable model from teacher outputs |
| [Instruction tuning](/concepts/learning/instruction-tuning) | aligning language-model behavior to task instructions |
| [Curriculum learning](/concepts/learning/curriculum-learning) | changing sample difficulty over training |
| [Imitation learning](/concepts/learning/imitation-learning) | learning from demonstrated actions |

## Self-Supervised Learning

| Method | Core Target |
| --- | --- |
| [Self-supervised learning](/concepts/learning/self-supervised-learning) | target signal derived from the input |
| [Masked modeling](/concepts/learning/masked-modeling) | reconstruct hidden tokens, patches, residues, atoms, or regions |
| [Contrastive learning](/concepts/learning/contrastive-learning) | pull positive views together and push negatives apart |
| [JEPA](/concepts/learning/jepa) | predict latent representation of a target view |
| [Augmentation policy](/concepts/learning/augmentation-policy) | define invariances through view construction |
| [Representation collapse](/concepts/learning/representation-collapse) | failure mode where embeddings lose useful variation |
| [Representation evaluation](/concepts/learning/representation-evaluation) | measure transfer beyond the pretraining loss |

## Reinforcement and Preference Learning

| Method | Core Object |
| --- | --- |
| [Reinforcement learning](/concepts/learning/reinforcement-learning) | policy, state, action, reward, return |
| [Policy gradient](/concepts/learning/policy-gradient) | gradient estimator for expected return |
| [Reward modeling](/concepts/learning/reward-modeling) | learned proxy for human, assay, or environment preference |
| [Preference optimization](/concepts/learning/preference-optimization) | pairwise or listwise preference signal |

## Objective Lens

Learning method note는 training signal을 명시해야 합니다.

$$
\theta^\star
=
\arg\min_\theta
\mathbb{E}_{(x,t)\sim q}
\left[
\mathcal{L}_\theta(x,t)
\right]
$$

where $q$ defines how examples and targets are sampled. For papers, this means the method is incomplete until the note states:

- example unit and split unit;
- target construction rule;
- loss and metric;
- augmentation, masking, negative sampling, reward, or preference source;
- evaluation task used to justify representation or behavior claims.

## Related

- [[concepts/tasks/index|Tasks]]
- [[concepts/architectures/index|Architectures]]
- [[concepts/evaluation/index|Evaluation]]
- [[agents/index|Agents]]
- [[molecular-modeling/protein-modeling|Protein modeling]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
