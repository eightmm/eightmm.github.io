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

## 핵심 노트

- [[concepts/learning/index|Learning methods]]
- [[concepts/learning/supervised-learning|Supervised learning]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/transfer-learning|Transfer learning]]
- [[concepts/learning/preference-optimization|Preference optimization]]

## 읽을 때 볼 질문

- label이 충분한가, 아니면 pretraining signal이 필요한가?
- representation을 instance-level, token-level, structure-level 중 어디에 맞출 것인가?
- downstream task와 pretraining task 사이에 mismatch가 있는가?
- objective가 downstream evaluation과 같은 정보를 요구하는가?

## Related

- [[ai/architectures|Architectures]]
- [[ai/evaluation|Evaluation]]
- [[ai/learning-methods|Learning methods]]
