---
title: Bayesian Inference
tags:
  - math
  - probability
  - inference
---

# Bayesian Inference

Bayesian inference updates uncertainty about a latent variable or parameter after observing data. It turns a prior and likelihood into a posterior distribution.

$$
p(\theta \mid \mathcal{D})
=
\frac{p(\mathcal{D}\mid \theta)p(\theta)}
{p(\mathcal{D})}
$$

where $\theta$ may be a model parameter, latent variable, hypothesis, structure, or hidden state.

## Core Objects

| Object | Meaning |
| --- | --- |
| Prior $p(\theta)$ | assumption before seeing the data |
| Likelihood $p(\mathcal{D}\mid\theta)$ | data model under a candidate $\theta$ |
| Evidence $p(\mathcal{D})$ | normalizing constant or marginal likelihood |
| Posterior $p(\theta\mid\mathcal{D})$ | updated uncertainty after observing data |
| Posterior predictive $p(y_\*\mid x_\*,\mathcal{D})$ | prediction averaged over posterior uncertainty |

The evidence integrates over possible parameters:

$$
p(\mathcal{D})
=
\int p(\mathcal{D}\mid\theta)p(\theta)\,d\theta
$$

## MAP and Regularization

Maximum a posteriori estimation chooses the posterior mode:

$$
\theta_{\mathrm{MAP}}
=
\arg\max_\theta
\left[
\log p(\mathcal{D}\mid\theta)
+
\log p(\theta)
\right]
$$

This is often equivalent to a regularized objective:

$$
\theta_{\mathrm{MAP}}
=
\arg\min_\theta
\left[
\mathcal{L}_{\mathrm{NLL}}(\theta)
+
\lambda \Omega(\theta)
\right]
$$

The prior becomes the regularizer. This interpretation is useful, but it does not automatically make the trained model a full Bayesian model.

## Prediction

Bayesian prediction averages over posterior uncertainty:

$$
p(y_\* \mid x_\*, \mathcal{D})
=
\int
p(y_\*\mid x_\*,\theta)
p(\theta\mid\mathcal{D})
d\theta
$$

Point-estimate models often replace this integral with one parameter value:

$$
p(y_\*\mid x_\*,\hat{\theta})
$$

That loses parameter uncertainty unless an approximation such as an ensemble, variational posterior, Laplace approximation, or Monte Carlo method is used.

## Approximation Routes

| Route | Core Idea | Check |
| --- | --- | --- |
| MAP | optimize posterior mode | no posterior spread by itself |
| Variational inference | fit $q_\phi(\theta)$ to approximate posterior | approximation family can be too narrow |
| Monte Carlo | sample from posterior or predictive distribution | chain quality or sampling budget matters |
| Laplace approximation | approximate posterior near a mode with curvature | local Gaussian assumption may fail |
| Ensembles | use multiple trained models as uncertainty proxy | diversity depends on training protocol |

## Why It Matters

- Uncertainty estimation should say what uncertainty is represented.
- Latent-variable models depend on posterior approximation.
- MAP estimation explains why priors and regularization are related.
- Posterior predictive distributions separate probability from hard decisions.
- Many "Bayesian" claims in papers are approximate and need an evidence boundary.

## Checks

- What is observed and what is latent?
- Is the prior explicit, learned, empirical, or only implicit?
- Is the posterior exact, approximate, amortized, or collapsed to a point estimate?
- Does uncertainty represent data noise, parameter uncertainty, model uncertainty, or all of them?
- Is the reported number a likelihood, posterior probability, MAP estimate, posterior mean, sample, or predictive probability?

## Related

- [[concepts/math/bayes-rule|Bayes rule]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]]
- [[concepts/generative-models/elbo|ELBO]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/calibration|Calibration]]
