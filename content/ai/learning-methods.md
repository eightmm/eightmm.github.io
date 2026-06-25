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

이 페이지는 한글 안내 페이지입니다. 링크된 `concepts/learning/*` 문서는 영어 canonical wiki note로 유지합니다.

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

## 핵심 노트

- [[concepts/learning/index|Learning methods]]
- [[concepts/learning/supervised-learning|Supervised learning]]
- [[concepts/learning/pretraining|Pretraining]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/augmentation-policy|Augmentation policy]]
- [[concepts/learning/representation-collapse|Representation collapse]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/knowledge-distillation|Knowledge distillation]]
- [[concepts/learning/instruction-tuning|Instruction tuning]]
- [[concepts/learning/transfer-learning|Transfer learning]]
- [[concepts/learning/domain-adaptation|Domain adaptation]]
- [[concepts/learning/curriculum-learning|Curriculum learning]]
- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[concepts/learning/policy-gradient|Policy gradient]]
- [[concepts/learning/reward-modeling|Reward modeling]]
- [[concepts/learning/imitation-learning|Imitation learning]]
- [[concepts/learning/active-learning|Active learning]]
- [[concepts/learning/preference-optimization|Preference optimization]]

## 읽을 때 볼 질문

- label이 충분한가, 아니면 pretraining signal이 필요한가?
- source domain과 target domain이 얼마나 다른가?
- representation을 instance-level, token-level, structure-level 중 어디에 맞출 것인가?
- adaptation이 pretraining 지식을 보존하는가, 아니면 target set에 과적합되는가?
- downstream task와 pretraining task 사이에 mismatch가 있는가?
- objective가 downstream evaluation과 같은 정보를 요구하는가?
- reward나 preference signal이 실제 목표와 proxy mismatch를 일으키지 않는가?

## Related

- [[ai/architectures|Architectures]]
- [[ai/evaluation|Evaluation]]
- [[ai/learning-methods|Learning methods]]
- [[agents/index|Agents]]
