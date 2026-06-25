---
title: Evaluation
tags:
  - ai
  - evaluation
---

# Evaluation

평가는 모델 목록을 실제 지식으로 바꾸는 기준입니다. AI note를 쓸 때는 어떤 benchmark에서 좋았는지보다, 어떤 split과 metric이 무엇을 검증하는지 먼저 봅니다.

이 페이지는 한글 안내 페이지입니다. 링크된 `concepts/evaluation/*` 문서는 영어 canonical wiki note로 유지합니다.

일반화 성능은 보지 못한 데이터 분포에서의 기대 손실로 보는 것이 기본입니다.

$$
R(f)
= \mathbb{E}_{(x,y)\sim p_{\mathrm{test}}}
\left[\mathcal{L}(f(x), y)\right]
$$

실험에서는 이를 finite test set 평균으로 추정합니다.

$$
\hat{R}(f)
= \frac{1}{m}\sum_{j=1}^{m}
\mathcal{L}(f(x_j), y_j)
$$

중요한 점은 $p_{\mathrm{test}}$가 실제로 알고 싶은 deployment distribution을 닮아야 한다는 것입니다.

## 핵심 노트

- [[concepts/evaluation/index|Evaluation]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]

## 읽을 때 볼 질문

- train/test split이 실제 generalization을 요구하는가?
- metric이 task objective와 맞는가?
- confidence나 calibration이 필요한 application인가?
- 실패를 data, model, optimization, evaluation 중 어디 문제로 분해할 수 있는가?

## Related

- [[ai/learning-methods|Learning methods]]
- [[bio-ai/index|Bio-AI]]
- [[papers/index|Papers]]
