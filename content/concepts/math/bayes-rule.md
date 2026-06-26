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

The same rule can be read as a proportionality:

$$
p(z\mid x)
\propto
p(x\mid z)p(z)
$$

because $p(x)$ does not depend on $z$. This is often the most useful view when comparing candidate latent states or parameters.

The evidence marginalizes over latent states:

$$
p(x)=\sum_z p(x\mid z)p(z)
$$

or, for continuous latent variables:

$$
p(x)=\int p(x\mid z)p(z)\,dz
$$

## Conditional Form

With additional context $c$, Bayes rule becomes:

$$
p(z\mid x,c)
=
\frac{p(x\mid z,c)p(z\mid c)}{p(x\mid c)}
$$

This is the form behind conditional generative models, prompt-conditioned prediction, and structure-conditioned molecular modeling.

## MAP Estimate

The maximum a posteriori estimate chooses the most likely latent state or parameter after observing data:

$$
z_{\mathrm{MAP}}
=
\arg\max_z p(z\mid x)
=
\arg\max_z
\left[
\log p(x\mid z)+\log p(z)
\right]
$$

The likelihood term rewards explaining the observation. The prior term encodes assumptions before seeing the observation.

## Posterior Predictive

Bayesian prediction averages over posterior uncertainty:

$$
p(y\mid x)
=
\int p(y\mid x,z)p(z\mid x)\,dz
$$

Practical ML systems often approximate this integral with ensembles, variational distributions, dropout approximations, or calibrated point estimates.

## Why It Matters

- Latent-variable models infer hidden causes from observations.
- Uncertainty estimation often asks for posterior distributions.
- Bayesian framing separates prior assumptions from evidence.
- MAP estimation connects priors to regularized optimization.
- Many practical ML systems approximate Bayesian reasoning without doing exact inference.
- Posterior uncertainty can be more informative than a single best prediction.

## Checks

- What is observed and what is latent?
- Is the prior explicit, learned, or implicit?
- Is posterior inference exact, approximate, amortized, or ignored?
- Does the uncertainty estimate reflect data uncertainty, model uncertainty, or both?
- Is the reported prediction a posterior mean, MAP estimate, sample, or calibrated score?

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/bayesian-inference|Bayesian inference]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/machine-learning/density-estimation|Density estimation]]
