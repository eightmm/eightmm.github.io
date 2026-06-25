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

## Practical Checks

- What distribution are samples drawn from?
- How many samples are used, and is the estimate stable?
- Are samples independent or correlated?
- Is the estimator biased by filtering, truncation, rejection, or selection?
- Is uncertainty reported with confidence intervals or repeated runs?

## Related

- [[concepts/math/expectation|Expectation]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
