---
title: Learning Methods
aliases:
  - research/self-supervised-learning
  - research/self-supervised-learning/index
tags:
  - ai
  - learning
---

# Learning Methods

학습 방법은 모델이 어떤 신호로 representation을 만드는지에 대한 분류입니다. 같은 architecture라도 supervised, self-supervised, preference-based objective에 따라 모델이 배우는 것이 달라집니다.

Supervised learning의 기본 objective는 label이 있는 데이터에서 prediction loss를 줄이는 것입니다.

$$
\min_\theta \mathbb{E}_{(x,y)\sim p_{\mathrm{data}}}
\left[\mathcal{L}(f_\theta(x), y)\right]
$$

Self-supervised learning은 label 대신 데이터 자체에서 target을 만듭니다. 예를 들어 masked modeling은 일부 token이나 feature를 가리고 복원합니다.

$$
\min_\theta \mathbb{E}_{x\sim p_{\mathrm{data}}}
\left[-\log p_\theta(x_{\mathrm{masked}} \mid x_{\mathrm{visible}})\right]
$$

Reinforcement learning은 label 대신 environment나 evaluator가 주는 reward를 최대화합니다.

$$
\max_\theta
J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T}\gamma^t r(s_t,a_t)
\right]
$$

Agent, tool-use, preference optimization을 읽을 때는 supervised fine-tuning과 RL-style objective를 분리해서 보는 것이 중요합니다.

## Route Map

| Route | Use For | Start |
| --- | --- | --- |
| Supervised signals | measured labels, classification, regression, target-conditioned prediction | [Supervised learning](/concepts/learning/supervised-learning), [Semi-supervised learning](/concepts/learning/semi-supervised-learning) |
| Pretraining signals | large unlabeled corpora, masked objectives, contrastive views, latent prediction | [Pretraining](/concepts/learning/pretraining), [Self-supervised learning](/concepts/learning/self-supervised-learning), [Masked modeling](/concepts/learning/masked-modeling) |
| Representation checks | collapse, probing, retrieval, transfer behavior | [Representation collapse](/concepts/learning/representation-collapse), [Representation evaluation](/concepts/learning/representation-evaluation), [Linear probing](/concepts/learning/linear-probing) |
| Contrastive and predictive SSL | positive/negative pairs, view construction, JEPA-style latent prediction | [Contrastive learning](/concepts/learning/contrastive-learning), [JEPA](/concepts/learning/jepa), [Augmentation policy](/concepts/learning/augmentation-policy) |
| Adaptation | fine-tuning, domain transfer, curriculum, distillation, instruction tuning | [Fine-tuning](/concepts/learning/fine-tuning), [Fine-tuning protocol](/concepts/learning/fine-tuning-protocol), [Transfer learning](/concepts/learning/transfer-learning), [Domain adaptation](/concepts/learning/domain-adaptation) |
| Feedback and control | preference data, reward models, policy optimization, imitation, active learning | [Preference optimization](/concepts/learning/preference-optimization), [Reinforcement learning](/concepts/learning/reinforcement-learning), [Reward modeling](/concepts/learning/reward-modeling) |

## 학습 신호 기준

| Method | Signal | Typical Objective | Read |
| --- | --- | --- | --- |
| Supervised learning | human or measured label $y$ | prediction loss $\mathcal{L}(f_\theta(x), y)$ | [Supervised learning](/concepts/learning/supervised-learning) |
| Semi-supervised learning | small labeled set plus unlabeled data | supervised loss plus consistency or pseudo-label term | [Semi-supervised learning](/concepts/learning/semi-supervised-learning) |
| Self-supervised learning | target derived from the input itself | masked, contrastive, predictive, or reconstruction loss | [Self-supervised learning](/concepts/learning/self-supervised-learning) |
| Contrastive learning | positive and negative pairs | rank matched pairs above mismatches | [Contrastive learning](/concepts/learning/contrastive-learning) |
| JEPA-style learning | representation prediction without pixel/token reconstruction | predict latent target from context | [JEPA](/concepts/learning/jepa) |
| Transfer / fine-tuning | pretrained model plus target task | adapt representation under a new validation rule | [Transfer learning](/concepts/learning/transfer-learning), [Fine-tuning](/concepts/learning/fine-tuning) |
| Preference optimization | pairwise or listwise preference signal | prefer chosen output over rejected output | [Preference optimization](/concepts/learning/preference-optimization) |
| Reinforcement learning | reward from environment or evaluator | maximize expected return | [Reinforcement learning](/concepts/learning/reinforcement-learning) |

## Contrastive and Preference Forms

Contrastive learning usually compares a positive pair against a set of negatives:

$$
\mathcal{L}_{\mathrm{NCE}}
=
-\log
\frac{\exp(\operatorname{sim}(z_i,z_i^+)/\tau)}
{\exp(\operatorname{sim}(z_i,z_i^+)/\tau)+
\sum_{j}\exp(\operatorname{sim}(z_i,z_j^-)/\tau)}
$$

Here $z_i$ is an anchor representation, $z_i^+$ is a positive representation, $z_j^-$ are negatives, and $\tau$ is a temperature.

Preference objectives compare outputs for the same context:

$$
\max_\theta
\mathbb{E}_{(x,y^+,y^-)}
\left[
\log \sigma
\left(
r_\theta(x,y^+) - r_\theta(x,y^-)
\right)
\right]
$$

Here $y^+$ is preferred over $y^-$ and $r_\theta$ is a learned or implicit reward score.

## 읽을 때 볼 질문

- label이 충분한가, 아니면 pretraining signal이 필요한가?
- source domain과 target domain이 얼마나 다른가?
- representation을 instance-level, token-level, structure-level 중 어디에 맞출 것인가?
- adaptation이 pretraining 지식을 보존하는가, 아니면 target set에 과적합되는가?
- downstream task와 pretraining task 사이에 mismatch가 있는가?
- representation 평가는 frozen probe, fine-tuning, retrieval 중 무엇으로 했는가?
- objective가 downstream evaluation과 같은 정보를 요구하는가?
- positive, negative, mask, context, target view가 downstream 의미를 보존하는가?
- pretraining corpus와 downstream test set 사이에 duplicate, scaffold, family, source leakage가 없는가?
- evaluation budget이 linear probe, kNN, full fine-tuning, low-data fine-tuning 중 무엇인가?
- reward나 preference signal이 실제 목표와 proxy mismatch를 일으키지 않는가?

## Related

- [[ai/architectures|Architectures]]
- [[ai/evaluation|Evaluation]]
- [[concepts/learning/index|Learning methods]]
- [[agents/index|Agents]]
