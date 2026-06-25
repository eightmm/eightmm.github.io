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

For a data distribution $p_{\mathrm{data}}$ and model $p_\theta$, maximum likelihood minimizes:

$$
H(p_{\mathrm{data}},p_\theta)
=
H(p_{\mathrm{data}})
+
D_{\mathrm{KL}}(p_{\mathrm{data}}\Vert p_\theta)
$$

Since $H(p_{\mathrm{data}})$ does not depend on $\theta$, minimizing cross-entropy minimizes the forward KL from data to model.

## Direction Matters

KL is not symmetric:

$$
D_{\mathrm{KL}}(p\Vert q)
\ne
D_{\mathrm{KL}}(q\Vert p)
$$

Forward KL tends to strongly penalize missing support where $p(x)>0$ and $q(x)$ is small. Reverse KL can prefer modes that avoid low-probability regions. This difference matters in variational inference, generative modeling, and policy optimization.

## Why It Matters

- Classification often minimizes cross-entropy.
- [[concepts/generative-models/vae|VAEs]] use KL regularization.
- Distribution shift can be described as a difference between training and deployment distributions.
- Preference optimization and policy learning often compare distributions.
- Calibration and uncertainty depend on interpreting probabilities, not only rankings.

## Checks

- Which distribution is the target and which is the model?
- Is the divergence direction $D_{\mathrm{KL}}(p\Vert q)$ or $D_{\mathrm{KL}}(q\Vert p)$?
- Are probabilities calibrated or only useful for ranking?
- Is the loss averaged over examples, tokens, or classes?
- Is the KL term a training objective, regularizer, diagnostic, or theoretical bound?

## Related

- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/learning/preference-optimization|Preference optimization]]
