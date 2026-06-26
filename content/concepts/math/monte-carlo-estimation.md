---
title: Monte Carlo Estimation
tags:
  - math
  - probability
  - estimation
---

# Monte Carlo Estimation

Monte Carlo estimation approximates expectations with samples. It appears in evaluation, simulation, generative modeling, reinforcement learning, uncertainty estimation, and approximate inference.

For a random variable $X\sim p(x)$ and function $f$, the target expectation is:

$$
\mu
=
\mathbb{E}_{X\sim p}[f(X)]
$$

With samples $x_1,\ldots,x_n\sim p$, the Monte Carlo estimator is:

$$
\hat{\mu}
=
\frac{1}{n}
\sum_{i=1}^{n}
f(x_i)
$$

The estimator variance decreases with sample count:

$$
\operatorname{Var}(\hat{\mu})
=
\frac{\operatorname{Var}(f(X))}{n}
$$

## Key Ideas

- Monte Carlo replaces an integral or expectation with an empirical average.
- More samples reduce variance but increase compute cost.
- Biased sampling requires correction if the goal is an expectation under another distribution.
- Seed, sample count, and sampling distribution are part of the experimental protocol.
- Many reported metrics are Monte Carlo estimates over data examples, seeds, generated samples, or rollouts.

## Estimation Contract

| Field | Question |
| --- | --- |
| Target expectation | What quantity is being estimated? |
| Sampling distribution | Where do samples come from? |
| Sample count | How many examples, seeds, rollouts, or generated samples? |
| Independence | Are samples independent, paired, batched, or correlated? |
| Filtering | Were failed or invalid samples removed? |
| Uncertainty | Is a confidence interval or standard error reported? |

The standard error for independent samples is:

$$
\operatorname{SE}(\hat{\mu})
=
\sqrt{\frac{\widehat{\operatorname{Var}}(f(X))}{n}}
$$

where:

$$
\widehat{\operatorname{Var}}(f(X))
=
\frac{1}{n-1}
\sum_{i=1}^{n}
\left(f(x_i)-\hat{\mu}\right)^2
$$

## Importance Sampling

If samples come from $q(x)$ but the target expectation is under $p(x)$:

$$
\mathbb{E}_{p}[f(X)]
=
\mathbb{E}_{q}
\left[
f(X)\frac{p(X)}{q(X)}
\right]
$$

The estimator is:

$$
\hat{\mu}_{\mathrm{IS}}
=
\frac{1}{n}
\sum_{i=1}^{n}
f(x_i)w_i,
\qquad
w_i=\frac{p(x_i)}{q(x_i)}
$$

Large or unstable weights can make the estimate high variance.

## AI and Computational Biology Uses

| Use | Samples | Common Trap |
| --- | --- | --- |
| generative model evaluation | generated samples | filtering invalid samples changes the denominator |
| diffusion or flow sampling | random seeds and solver paths | sample quality depends on step budget |
| uncertainty estimation | model samples or posterior samples | correlated samples overstate certainty |
| RL or agent evaluation | rollouts | few tasks or seeds give noisy estimates |
| virtual screening | candidate library draws or top-K subsets | active prevalence changes enrichment |
| molecular simulation | trajectory frames | autocorrelation reduces effective sample size |

For correlated samples, the effective sample size can be much smaller than the raw sample count.

## Practical Checks

- What distribution are samples drawn from?
- How many samples are used, and is the estimate stable?
- Are samples independent or correlated?
- Is the estimator biased by filtering, truncation, rejection, or selection?
- Is uncertainty reported with confidence intervals or repeated runs?
- Are random seeds and sampling budgets fixed across methods?
- Are failed generations, failed simulations, or invalid outputs part of the reported denominator?

## Related

- [[concepts/math/expectation|Expectation]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/bootstrap-evaluation|Bootstrap evaluation]]
- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
