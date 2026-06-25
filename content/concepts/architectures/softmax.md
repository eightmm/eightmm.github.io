---
title: Softmax
tags:
  - architectures
  - machine-learning
---

# Softmax

Softmax converts logits into a probability distribution over mutually exclusive choices.

$$
p_i
= \operatorname{softmax}(z)_i
= \frac{\exp(z_i)}{\sum_{j=1}^{K}\exp(z_j)}
$$

Here $z_i$ is a logit and $p_i$ is the normalized probability for class or option $i$.

Softmax is invariant to adding the same constant to all logits:

$$
\operatorname{softmax}(z)_i
=
\operatorname{softmax}(z-c\mathbf{1})_i
$$

This is used for numerical stability:

$$
p_i
=
\frac{\exp(z_i-\max_j z_j)}
{\sum_{j=1}^{K}\exp(z_j-\max_j z_j)}
$$

## Temperature

A temperature rescales logits:

$$
p_i(\tau)
=
\frac{\exp(z_i/\tau)}
{\sum_{j=1}^{K}\exp(z_j/\tau)}
$$

Small $\tau$ makes the distribution sharper. Large $\tau$ makes it flatter. Temperature is used in decoding, calibration, distillation, and contrastive losses.

## Cross-Entropy Link

With one-hot target $y$, softmax cross-entropy is:

$$
\mathcal{L}
=
-\sum_{i=1}^{K} y_i \log p_i
$$

For logits $z$ and probabilities $p=\operatorname{softmax}(z)$:

$$
\frac{\partial \mathcal{L}}{\partial z_i}
=
p_i-y_i
$$

This simple gradient is one reason softmax and cross-entropy are usually implemented together.

## Masked Softmax

For attention or constrained choices, invalid positions are masked before softmax:

$$
p_i
=
\frac{\exp(z_i + m_i)}
{\sum_j \exp(z_j + m_j)}
$$

where $m_i=0$ for valid positions and $m_i=-\infty$ for invalid positions.

## Where It Appears

- Attention weights in [[concepts/architectures/attention|Attention]].
- Classification heads.
- Autoregressive next-token distributions.
- Routing probabilities in [[concepts/architectures/mixture-of-experts|Mixture of experts]].

## Checks

- Softmax is sensitive to logit scale.
- Masks usually add $-\infty$ to impossible positions before softmax.
- For multi-label prediction, sigmoid is usually more appropriate than softmax.
- Probabilities can still be poorly calibrated; see [[concepts/evaluation/calibration|Calibration]].
- Check the axis: class dimension, token dimension, expert dimension, or candidate list.
- Check whether logits are raw scores, normalized similarities, or temperature-scaled scores.

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/machine-learning/cross-entropy-loss|Cross-entropy loss]]
- [[concepts/machine-learning/probabilistic-prediction|Probabilistic prediction]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/evaluation/calibration|Calibration]]
