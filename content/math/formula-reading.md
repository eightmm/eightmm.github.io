---
title: Formula Reading
aliases:
  - math/formula-reading-for-ai
tags:
  - math
  - formula
---

# Formula Reading

논문 수식은 notation 자체보다 어떤 object, distribution, objective, estimator, metric을 고정하는지가 중요합니다. 수식이 나오면 먼저 모양을 단순화하고, 그 식이 주장하는 claim을 확인합니다.

$$
\widehat{Q}
=
\frac{1}{n}\sum_{i=1}^{n} q(x_i, y_i, f_\theta)
\qquad
\text{estimates}
\qquad
Q = \mathbb{E}_{(x,y)\sim p} q(x,y,f_\theta)
$$

첫 질문은 “이 수식이 어려운가”가 아니라 “이 수식이 어떤 quantity를 정의하는가”입니다. 정의식, objective, estimator, metric, algorithm update를 구분하면 대부분의 논문 수식은 읽기 쉬워집니다.

## Reading Order

| Step | Ask | Route |
| --- | --- | --- |
| Type | scalar, vector, matrix, tensor, graph, coordinate, distribution 중 무엇인가? | [Tensor shape notation](/concepts/math/tensor-shape-notation) |
| Axis | batch, token, residue, atom, node, head, coordinate, time 축은 무엇인가? | [Linear Algebra](/math/linear-algebra) |
| Distribution | expectation이나 sampling이 data, model, noise, policy, test distribution 중 어디서 오나? | [Probability and Statistics](/math/probability-statistics) |
| Parameter | 미분 대상이 parameter $\theta$, input $x$, coordinate $X$, time $t$ 중 무엇인가? | [Calculus and Gradients](/math/calculus-gradients) |
| Objective | loss, likelihood, score, reward, energy, constraint 중 무엇인가? | [Information and Likelihood](/math/information-likelihood) |
| Estimate | population quantity와 finite-sample estimate가 분리되어 있는가? | [Estimator vs metric](/concepts/math/estimator-vs-metric) |
| Claim | 이 식이 architecture, training, generation, evaluation, domain workflow 중 무엇을 뒷받침하는가? | [Model Reading Map](/ai/model-reading-map) |

## Formula Type

| Type | 형태 | 읽는 법 |
| --- | --- | --- |
| Definition | $z=h_\theta(x)$ | symbol과 shape를 고정 |
| Objective | $\min_\theta \mathcal{L}(\theta)$ | 무엇을 낮추는지 확인 |
| Estimator | $\hat{Q}=\frac{1}{n}\sum_i q_i$ | finite sample과 population 구분 |
| Update | $\theta_{t+1}=\theta_t-\eta\nabla_\theta \mathcal{L}$ | state transition으로 읽기 |
| Constraint | $g(x)\le 0$ 또는 $x\in\mathcal{S}$ | feasible set 확인 |
| Metric | $m(\hat{y},y)$ | claim과 metric alignment 확인 |

## Formula by Claim Type

| Claim | Formula pattern | Check |
| --- | --- | --- |
| Prediction | $\hat{y}=f_\theta(x)$ | $x$, $y$, output space가 명확한가? |
| ERM | $\min_\theta \frac{1}{n}\sum_i \mathcal{L}(f_\theta(x_i), y_i)$ | sample unit과 split unit이 같은가? |
| Likelihood | $\max_\theta \sum_i \log p_\theta(x_i)$ | modeled distribution이 무엇인가? |
| Contrastive | $-\log \frac{\exp(s(z,z^+)/\tau)}{\sum_j \exp(s(z,z_j)/\tau)}$ | positive/negative pair가 claim과 맞는가? |
| Denoising | $\mathbb{E}_{t,\epsilon}\|\epsilon-\epsilon_\theta(x_t,t)\|^2$ | noise process와 target이 명확한가? |
| Flow matching | $\mathbb{E}_{t,x_t}\|v_\theta(x_t,t)-u_t(x_t)\|^2$ | velocity field와 sampling path가 분리되는가? |
| Evaluation | $\hat{m}=\frac{1}{m}\sum_j g(\hat{y}_j,y_j)$ | metric과 confidence interval이 있는가? |

## Local Rewrite

수식이 복잡하면 먼저 아래처럼 지역 notation으로 바꿉니다.

| Symbol | Meaning |
| --- | --- |
| $u_i$ | one example unit |
| $x_i$ | observed input or representation |
| $y_i$ | label, target, preference, reward, or reference output |
| $z_i=h_\theta(x_i)$ | learned representation |
| $\hat{y}_i=f_\theta(x_i)$ | model output |
| $\mathcal{L}_i$ | per-example training loss |
| $m_i$ | per-example evaluation contribution |

## Shape Check

수식이 맞아 보이는지 빠르게 확인하려면 shape를 씁니다.

$$
X \in \mathbb{R}^{B \times N \times d},
\qquad
W_Q \in \mathbb{R}^{d \times d_h},
\qquad
Q = XW_Q \in \mathbb{R}^{B \times N \times d_h}
$$

행렬곱, sum, expectation, norm의 결과 shape가 claim과 맞지 않으면 notation이 생략되었거나 수식 이해가 틀린 것입니다.

## Related

- [[math/index|Math]]
- [[math/evaluation-math|Evaluation Math]]
- [[concepts/math/estimator-vs-metric|Estimator vs metric]]
- [[ai/model-reading-map|Model Reading Map]]
