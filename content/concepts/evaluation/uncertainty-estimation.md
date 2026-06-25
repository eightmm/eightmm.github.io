---
title: Uncertainty Estimation
tags:
  - evaluation
  - uncertainty
  - methodology
---

# Uncertainty Estimation

Uncertainty estimation asks how much trust to place in a prediction. It is different from raw confidence: a model can output high confidence while being wrong or out of distribution.

For a predictive distribution:

$$
p_\theta(y\mid x)
$$

a point prediction hides uncertainty, while the full distribution can expose ambiguity:

$$
\hat{y} = \arg\max_y p_\theta(y\mid x)
$$

## Types

- Aleatoric uncertainty: noise or ambiguity in the data itself.
- Epistemic uncertainty: uncertainty from limited data or model knowledge.
- Distributional uncertainty: uncertainty caused by shift from training data.

For regression with Gaussian likelihood:

$$
y \sim \mathcal{N}(\mu_\theta(x), \sigma_\theta^2(x))
$$

the model predicts both mean and variance.

## Checks

- Is uncertainty used for ranking, abstention, active learning, or decision-making?
- Is uncertainty calibrated on held-out data?
- Does uncertainty increase under out-of-distribution inputs?
- Are epistemic and aleatoric uncertainty conflated?
- Is the uncertainty estimate evaluated against downstream risk?

## Related

- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[concepts/math/bayes-rule|Bayes rule]]
- [[concepts/machine-learning/regression|Regression]]
