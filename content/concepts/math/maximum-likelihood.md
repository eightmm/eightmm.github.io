---
title: Maximum Likelihood
tags:
  - math
  - likelihood
  - machine-learning
---

# Maximum Likelihood

Maximum likelihood chooses model parameters that make the observed data likely under the model.

For independent examples:

$$
\theta^\star
= \arg\max_\theta
\prod_{i=1}^{n} p_\theta(x_i)
$$

It is usually optimized as log-likelihood:

$$
\theta^\star
= \arg\max_\theta
\sum_{i=1}^{n}\log p_\theta(x_i)
$$

or negative log-likelihood:

$$
\mathcal{L}_{\mathrm{NLL}}
= -\sum_{i=1}^{n}\log p_\theta(x_i)
$$

As an empirical expectation:

$$
\mathcal{L}_{\mathrm{NLL}}
=
-n\,
\mathbb{E}_{x\sim \hat{p}_{\mathcal{D}}}
\left[
\log p_\theta(x)
\right]
$$

where $\hat{p}_{\mathcal{D}}$ is the empirical data distribution.

For supervised learning, the conditional version is:

$$
\mathcal{L}_{\mathrm{NLL}}
= -\sum_{i=1}^{n}\log p_\theta(y_i\mid x_i)
$$

This can be written as an empirical risk:

$$
\hat{R}_{\mathrm{NLL}}(\theta)
=
\mathbb{E}_{(x,y)\sim \hat{p}_{\mathcal{D}}}
\left[
-\log p_\theta(y\mid x)
\right]
$$

where $\hat{p}_{\mathcal{D}}$ is the empirical distribution induced by the dataset and sampling rule.

For sequence modeling with the chain rule:

$$
-\log p_\theta(x_{1:T})
=
-\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t})
$$

If sequences have different lengths, the denominator is part of the claim:

$$
\mathrm{NLL}_{\mathrm{token}}
=
-
\frac{\sum_i\sum_{t\in \mathcal{V}_i}
\log p_\theta(x_{i,t}\mid x_{i,<t})}
{\sum_i |\mathcal{V}_i|}
$$

where $\mathcal{V}_i$ is the set of valid non-padding positions. Token-average and sequence-average likelihood optimize different populations.

## Why It Matters

- Cross-entropy classification is conditional maximum likelihood.
- Autoregressive language modeling minimizes next-token negative log-likelihood.
- Density estimation and many generative models are likelihood-based.
- Likelihood is not always aligned with sample quality or downstream utility.
- Maximizing likelihood is equivalent to minimizing cross-entropy from the data distribution to the model distribution.

## Likelihood Claim Map

| Claim | Likelihood Form | Watch |
| --- | --- | --- |
| classifier probability | $p_\theta(y\mid x)$ | calibration, class prevalence, label noise |
| autoregressive model | $\prod_t p_\theta(x_t\mid x_{<t})$ | tokenization, length normalization, teacher forcing |
| density model | $p_\theta(x)$ | exact vs approximate normalization |
| latent-variable model | $\int p_\theta(x,z)\,dz$ or ELBO | bound tightness and posterior approximation |
| conditional generation | $p_\theta(x\mid c)$ | condition leakage and sampling metric |
| regression as likelihood | $p_\theta(y\mid x)$ with Gaussian/Laplace/etc. | noise model, target scale, censoring |

## Likelihood vs Utility

Likelihood is a proper training objective for probability models, but it may not be the final utility:

| Setting | Likelihood Can Miss |
| --- | --- |
| generation | sample validity, novelty, diversity, and constraint satisfaction |
| ranking | top-k retrieval or enrichment under a candidate corpus |
| molecular modeling | chemical validity, assay utility, pose geometry, or downstream screening value |
| long sequences | sequence-level success when token-level loss is dominated by easy positions |
| imbalanced classification | decision utility under deployment prevalence |

## Checks

- What probability is being maximized: $p(x)$, $p(y\mid x)$, or $p(x,y)$?
- Is the likelihood exact, approximated, bounded, or implicit?
- Does high likelihood correspond to the task metric?
- Are examples assumed independent when they are grouped or duplicated?
- Is the objective token-level, example-level, trajectory-level, or structure-level?
- What is the averaging denominator: example, token, atom, residue, edge, pair, trajectory, or batch?
- Is model selection based on likelihood, downstream metric, or a separate validation utility?
- Does the reported likelihood use the same preprocessing, tokenization, masking, and support as training?

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/machine-learning/density-estimation|Density estimation]]
- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/machine-learning/objective-metric-alignment|Objective-metric alignment]]
- [[concepts/evaluation/proper-scoring-rule|Proper scoring rule]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
