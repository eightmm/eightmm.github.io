---
title: Bayes Rule
tags:
  - math
  - probability
---

# Bayes Rule

Bayes rule relates prior belief, likelihood, evidence, and posterior belief.

$$
p(z\mid x)
= \frac{p(x\mid z)p(z)}{p(x)}
$$

where:

- $p(z)$ is the prior.
- $p(x\mid z)$ is the likelihood.
- $p(x)$ is the evidence or marginal likelihood.
- $p(z\mid x)$ is the posterior.

The evidence marginalizes over latent states:

$$
p(x)=\sum_z p(x\mid z)p(z)
$$

or, for continuous latent variables:

$$
p(x)=\int p(x\mid z)p(z)\,dz
$$

## Why It Matters

- Latent-variable models infer hidden causes from observations.
- Uncertainty estimation often asks for posterior distributions.
- Bayesian framing separates prior assumptions from evidence.
- Many practical ML systems approximate Bayesian reasoning without doing exact inference.

## Checks

- What is observed and what is latent?
- Is the prior explicit, learned, or implicit?
- Is posterior inference exact, approximate, amortized, or ignored?
- Does the uncertainty estimate reflect data uncertainty, model uncertainty, or both?

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/machine-learning/density-estimation|Density estimation]]
