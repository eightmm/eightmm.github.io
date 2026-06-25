---
title: Probability Metrics
tags:
  - evaluation
  - metrics
  - calibration
---

# Probability Metrics

Probability metrics evaluate [[concepts/machine-learning/probabilistic-prediction|predicted probabilities]], not only hard decisions or rankings. They matter when a model's confidence drives triage, filtering, decision thresholds, uncertainty estimates, or downstream planning.

For predictions $p_\theta(y\mid x)$ on test examples $(x_i,y_i)$, negative log-likelihood is:

$$
\operatorname{NLL}
=
-
\frac{1}{n}
\sum_{i=1}^{n}
\log p_\theta(y_i\mid x_i)
$$

This is the same mathematical form as a [[concepts/machine-learning/negative-log-likelihood|negative log-likelihood]] training loss, but in evaluation it is computed on held-out data under a fixed protocol.

## Brier Score

For binary labels $y_i\in\{0,1\}$ and predicted probability $p_i=p_\theta(y=1\mid x_i)$:

$$
\operatorname{Brier}
=
\frac{1}{n}
\sum_{i=1}^{n}
(p_i-y_i)^2
$$

For multiclass prediction:

$$
\operatorname{Brier}
=
\frac{1}{n}
\sum_{i=1}^{n}
\sum_{k=1}^{K}
(p_{ik}-y_{ik})^2
$$

where $p_{ik}$ is the predicted probability for class $k$ and $y_{ik}$ is the one-hot or soft target.

## Proper Scoring Rule

A proper scoring rule rewards honest probabilities: the expected score is optimized when the predicted distribution equals the true conditional distribution.

For a scoring rule $S(p,y)$, propriety means:

$$
\mathbb{E}_{Y\sim q}
[S(q,Y)]
\le
\mathbb{E}_{Y\sim q}
[S(p,Y)]
$$

for a loss-style score where lower is better. NLL and Brier score are common proper scoring rules.

## Metric, Loss, and Calibration

- Loss: used to update parameters during training.
- Probability metric: computed on validation or test data to evaluate probabilistic predictions.
- [[concepts/evaluation/calibration|Calibration]]: asks whether predicted confidence matches empirical frequency.
- [[concepts/evaluation/threshold-selection|Threshold selection]]: converts probabilities into decisions.

A model can have good accuracy but poor NLL if it is overconfident on errors. It can also have a strong ranking metric but unreliable probabilities.

## Checks

- Are probabilities required, or are rankings/classes enough?
- Is NLL averaged per example, token, atom, sequence, or graph?
- Are padding and invalid positions masked consistently?
- Are probabilities calibrated on validation data, not the test set?
- Are threshold-dependent metrics reported separately from probability metrics?

## Related

- [[concepts/evaluation/metric|Metric]]
- [[concepts/machine-learning/probabilistic-prediction|Probabilistic prediction]]
- [[concepts/machine-learning/decision-rule|Decision rule]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/reliability-diagram|Reliability diagram]]
- [[concepts/evaluation/threshold-selection|Threshold selection]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/machine-learning/cross-entropy-loss|Cross-entropy loss]]
