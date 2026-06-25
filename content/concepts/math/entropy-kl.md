---
title: Entropy and KL Divergence
tags:
  - math
  - information-theory
  - probability
---

# Entropy and KL Divergence

Entropy measures uncertainty in a distribution. KL divergence measures how much one distribution differs from another.

Entropy:

$$
H(p) = -\sum_x p(x)\log p(x)
$$

Cross-entropy:

$$
H(p,q) = -\sum_x p(x)\log q(x)
$$

KL divergence:

$$
D_{\mathrm{KL}}(p\Vert q)
= \sum_x p(x)\log\frac{p(x)}{q(x)}
$$

These are related:

$$
H(p,q)=H(p)+D_{\mathrm{KL}}(p\Vert q)
$$

## Why It Matters

- Classification often minimizes cross-entropy.
- [[concepts/generative-models/vae|VAEs]] use KL regularization.
- Distribution shift can be described as a difference between training and deployment distributions.
- Preference optimization and policy learning often compare distributions.

## Checks

- Which distribution is the target and which is the model?
- Is the divergence direction $D_{\mathrm{KL}}(p\Vert q)$ or $D_{\mathrm{KL}}(q\Vert p)$?
- Are probabilities calibrated or only useful for ranking?
- Is the loss averaged over examples, tokens, or classes?

## Related

- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/learning/preference-optimization|Preference optimization]]
