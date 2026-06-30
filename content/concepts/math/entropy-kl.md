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

Equivalently:

$$
D_{\mathrm{KL}}(p\Vert q)
=
\mathbb{E}_{x\sim p}
\left[
\log p(x) - \log q(x)
\right]
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

For a finite dataset $\{x_i\}_{i=1}^{n}$, the empirical negative log-likelihood is:

$$
\widehat{H}(p_{\mathrm{data}},p_\theta)
=
-\frac{1}{n}\sum_{i=1}^{n}\log p_\theta(x_i)
$$

For token models, the same idea is usually averaged over tokens:

$$
\mathcal{L}_{\mathrm{NLL}}
=
-\frac{1}{\sum_i T_i}
\sum_i \sum_{t=1}^{T_i}
\log p_\theta(x_{i,t}\mid x_{i,<t})
$$

This is why comparing language-model losses requires knowing whether the loss is averaged per sequence, token, byte, or example.

## Direction Matters

KL is not symmetric:

$$
D_{\mathrm{KL}}(p\Vert q)
\ne
D_{\mathrm{KL}}(q\Vert p)
$$

Forward KL tends to strongly penalize missing support where $p(x)>0$ and $q(x)$ is small. Reverse KL can prefer modes that avoid low-probability regions. This difference matters in variational inference, generative modeling, and policy optimization.

## Support Conditions

KL is finite only when $q(x)>0$ wherever $p(x)>0$:

$$
p(x)>0 \Rightarrow q(x)>0
$$

If the model assigns zero probability to an event that the target distribution can produce, $D_{\mathrm{KL}}(p\Vert q)$ becomes infinite. In implementations, smoothing, clipping, or finite precision can hide this issue, but the modeling assumption is still there.

## Continuous Variables

For continuous variables, sums become integrals:

$$
D_{\mathrm{KL}}(p\Vert q)
=
\int p(x)\log\frac{p(x)}{q(x)}\,dx
$$

Differential entropy can be negative and changes under reparameterization, so it should not be interpreted exactly like discrete entropy. KL divergence remains invariant to smooth invertible reparameterizations.

## Common Objective Patterns

| Pattern | Formula Shape | Meaning |
| --- | --- | --- |
| classification cross-entropy | $-\sum_c y_c\log \hat{p}_c$ | negative log probability of the target class |
| maximum likelihood | $-\mathbb{E}_{p_{\mathrm{data}}}\log p_\theta(x)$ | fit model density to data |
| VAE regularization | $D_{\mathrm{KL}}(q_\phi(z\mid x)\Vert p(z))$ | keep approximate posterior near prior |
| distillation | $H(p_{\mathrm{teacher}},p_{\mathrm{student}})$ | match teacher distribution |
| policy regularization | $D_{\mathrm{KL}}(\pi_{\mathrm{new}}\Vert \pi_{\mathrm{old}})$ or reverse | control policy drift |

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
- Is support mismatch possible?
- Is the entropy discrete or differential?
- Is the reported value normalized per token, per sequence, per dimension, or per example?

## Related

- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/cross-entropy-loss|Cross-entropy loss]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/learning/preference-optimization|Preference optimization]]
