---
title: Numerical Computing
tags:
  - math
  - numerical-computing
  - stability
---

# Numerical Computing

Numerical computing은 finite-precision computation의 수학입니다. AI formula는 real number 위에서 쓰이지만, 실제 training과 inference는 floating-point tensor, limited memory, finite accumulation order 위에서 실행됩니다.

이 페이지는 algebraically equivalent한 formula가 hardware 위에서 왜 다르게 동작할 수 있는지 설명하므로 Math에 둡니다. System-specific tradeoff는 [[infra/gpu/index|GPU infra]], [[concepts/systems/training-run|Training run]], [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]와 연결됩니다.

## Route Map

| Route | Use for | Start |
| --- | --- | --- |
| Stable probabilities | softmax, log-sum-exp, likelihood, attention logit | [Softmax](/concepts/architectures/softmax), [Information and likelihood](/math/information-likelihood) |
| Stable activations | normalization, residual scaling, exploding/vanishing value | [Normalization](/concepts/architectures/normalization), [Training stability](/concepts/machine-learning/training-stability) |
| Stable gradients | clipping, accumulation, underflow, optimizer sensitivity | [Gradient clipping](/concepts/machine-learning/gradient-clipping), [Gradient accumulation](/concepts/machine-learning/gradient-accumulation), [Calculus and gradients](/math/calculus-gradients) |
| Precision and memory | mixed precision, reduction, tensor layout, memory-compute tradeoff | [Memory-compute tradeoff](/concepts/systems/memory-compute-tradeoff), [GPU infra](/infra/gpu), [Training run](/concepts/systems/training-run) |
| Debugging boundary | 문제가 mathematical, numerical, model, optimizer, system behavior 중 어디에 있는가 | [Infra](/infra), [Evaluation](/ai/evaluation) |

## Floating Point

Floating-point number는 real number를 finite precision으로 근사합니다.

$$
\operatorname{fl}(x)
=
x(1+\delta),
\quad
|\delta| \le \epsilon
$$

여기서 $\epsilon$은 machine-dependent precision scale입니다. 더 작은 precision은 memory와 bandwidth를 아끼지만 rounding sensitivity를 키웁니다.

## Overflow and Underflow

Exponential은 큰 positive input에서 overflow하고, 큰 negative input에서 underflow할 수 있습니다.

$$
\exp(x)
\to
\infty
\quad
\text{or}
\quad
0
$$

이는 softmax, likelihood, attention, contrastive learning, energy-based scoring에서 중요합니다.

## Log-Sum-Exp

Stable log-sum-exp trick은 exponentiation 전에 maximum을 빼는 방식입니다.

$$
\operatorname{logsumexp}(x)
=
\log \sum_i \exp(x_i)
=
m + \log \sum_i \exp(x_i - m),
\quad
m = \max_i x_i
$$

이렇게 하면 가장 큰 exponent가 $\exp(0)=1$로 유지됩니다.

## Stable Softmax

Softmax는 보통 아래처럼 계산하는 것이 안전합니다.

$$
\operatorname{softmax}(x)_i
=
\frac{\exp(x_i-m)}
{\sum_j \exp(x_j-m)},
\quad
m=\max_j x_j
$$

이는 ordinary softmax와 수학적으로 같지만 numerically safer합니다.

## Conditioning

작은 input change가 큰 output change를 만들 수 있으면 problem이 ill-conditioned라고 봅니다. Matrix $A$의 condition number는 아래와 같습니다.

$$
\kappa(A)
=
\|A\| \|A^{-1}\|
$$

큰 $\kappa(A)$는 system solving, matrix inversion, gradient propagation이 noise와 rounding에 민감할 수 있음을 뜻합니다.

## Precision in Training

Mixed precision은 memory와 throughput을 바꾸지만 numerical behavior도 함께 바꿉니다. 흔한 risk point는 아래와 같습니다.

- tiny gradients underflowing to zero
- large activations or logits overflowing
- reductions accumulating in a different order
- normalization statistics losing precision
- optimizer state requiring more precision than activations

Loss scaling은 gradient underflow를 줄이는 방법 중 하나입니다.

$$
\nabla_\theta (s\mathcal{L})
=
s\nabla_\theta \mathcal{L}
$$

Scaled gradient는 optimizer update 전에 다시 unscale됩니다.

## Reduction Order

Floating-point addition은 완전히 associative하지 않습니다.

$$
(a+b)+c
\neq
a+(b+c)
$$

따라서 mathematical expression이 같아도 parallel reduction, distributed training, 다른 kernel은 조금 다른 결과를 만들 수 있습니다.

## Checks

- logit, loss, probability가 stable formula로 계산되는가?
- reduction이 dynamic range에 맞는 precision에서 수행되는가?
- instability가 overflow, underflow, NaN propagation, gradient explosion 중 무엇에서 오는가?
- 문제가 mathematical, numerical, architectural, optimizer-related, hardware-related 중 어디에 있는가?
- deterministic expectation이 해당 kernel과 distributed setup에서 현실적인가?

## Related

- [[math/index|Math]]
- [[math/calculus-gradients|Calculus and gradients]]
- [[concepts/architectures/softmax|Softmax]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
