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

For classification, predictive entropy is a simple uncertainty score:

$$
H[p_\theta(y\mid x)]
=
-\sum_{k=1}^{K}
p_\theta(y=k\mid x)
\log p_\theta(y=k\mid x)
$$

High entropy means the predictive distribution is diffuse, but it does not by itself prove the model knows when it is wrong.

## Evaluation

Uncertainty should be evaluated against a decision:

- Abstention: are uncertain examples more likely to be wrong?
- Selective prediction: does risk decrease as the system accepts fewer examples?
- Active learning: do queried examples improve the model efficiently?
- OOD detection: does uncertainty increase under shift?
- Risk ranking: do high-risk predictions receive larger uncertainty?
- Calibration: do confidence levels match empirical correctness?

## Checks

- Is uncertainty used for ranking, abstention, active learning, or decision-making?
- Are coverage and selective risk reported if the system may abstain?
- Is uncertainty calibrated on held-out data?
- Does uncertainty increase under out-of-distribution inputs?
- Are epistemic and aleatoric uncertainty conflated?
- Is the uncertainty estimate evaluated against downstream risk?

## Related

- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/selective-prediction|Selective prediction]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[concepts/math/bayes-rule|Bayes rule]]
- [[concepts/machine-learning/regression|Regression]]
