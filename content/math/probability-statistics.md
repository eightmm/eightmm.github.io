---
title: Probability and Statistics
tags:
  - math
  - probability
  - statistics
---

# Probability and Statistics

Probability describes uncertainty and data-generating processes. Statistics describes how finite samples estimate unknown quantities.

$$
\mathbb{E}_{(x,y)\sim p_{\mathrm{data}}}
[\mathcal{L}(f(x), y)]
$$

The distribution under the expectation matters as much as the loss itself.

## Probability

- [[concepts/math/random-variable|Random variable]]
- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/normal-distribution|Normal distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/bayes-rule|Bayes rule]]

## Statistics

- [[concepts/math/statistical-estimator|Statistical estimator]]
- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/math/central-limit-theorem|Central limit theorem]]
- [[concepts/math/hypothesis-testing|Hypothesis testing]]
- [[concepts/math/bias-variance-tradeoff|Bias-variance tradeoff]]
- [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]]

## AI Connections

- Probabilistic prediction needs calibrated probabilities, not only scores.
- Dataset shift changes the distribution under the expected risk.
- Uncertainty estimation depends on what randomness is being modeled.
- Hypothesis testing and confidence intervals help interpret benchmark differences.

## Checks

- Is the probability conditional or marginal?
- Is the estimate biased, high-variance, or data-leaking?
- Is the test distribution the same as the deployment distribution?
- Are repeated evaluations creating multiple-comparison risk?

## Related

- [[math/index|Math]]
- [[ai/evaluation|Evaluation]]
- [[concepts/evaluation/calibration|Calibration]]
