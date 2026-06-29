---
title: Formula Reading for AI
tags:
  - math
  - formula
  - ai
---

# Formula Reading for AI

AI 논문 수식은 notation 자체보다 어떤 object, distribution, objective, estimator, metric을 고정하는지가 중요합니다. 수식이 나오면 먼저 모양을 단순화하고, 그 식이 주장하는 claim을 확인합니다.

$$
\widehat{Q}
=
\frac{1}{n}\sum_{i=1}^{n} q(x_i, y_i, f_\theta)
\qquad
\text{estimates}
\qquad
Q = \mathbb{E}_{(x,y)\sim p} q(x,y,f_\theta)
$$

## Reading Order

| Step | Ask | Route |
| --- | --- | --- |
| Type | scalar, vector, matrix, tensor, graph, coordinate, distribution 중 무엇인가? | [[concepts/math/tensor-shape-notation|Tensor shape notation]] |
| Axis | batch, token, residue, atom, node, head, coordinate, time 축은 무엇인가? | [[math/linear-algebra|Linear Algebra]] |
| Distribution | expectation이나 sampling이 data, model, noise, policy, test distribution 중 어디서 오나? | [[math/probability-statistics|Probability and Statistics]] |
| Parameter | 미분 대상이 parameter $\theta$, input $x$, coordinate $X$, time $t$ 중 무엇인가? | [[math/calculus-gradients|Calculus and Gradients]] |
| Objective | loss, likelihood, score, reward, energy, constraint 중 무엇인가? | [[math/information-likelihood|Information and Likelihood]] |
| Estimate | population quantity와 finite-sample estimate가 분리되어 있는가? | [[concepts/math/estimator-vs-metric|Estimator vs metric]] |
| Claim | 이 식이 architecture, training, generation, evaluation 중 무엇을 뒷받침하는가? | [[ai/model-reading-map|Model Reading Map]] |

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

## Related

- [[math/index|Math]]
- [[math/evaluation-math|Evaluation Math]]
- [[concepts/math/estimator-vs-metric|Estimator vs metric]]
- [[ai/model-reading-map|Model Reading Map]]
