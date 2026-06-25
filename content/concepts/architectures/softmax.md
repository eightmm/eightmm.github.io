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

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/evaluation/calibration|Calibration]]
