---
title: Probabilistic Prediction
tags:
  - machine-learning
  - probability
  - evaluation
---

# Probabilistic Prediction

Probabilistic prediction returns a distribution over possible outputs instead of only a point prediction. It matters when confidence, calibration, uncertainty, abstention, ranking, or decision cost affects the downstream use.

For supervised learning, a probabilistic model estimates:

$$
p_\theta(y\mid x)
$$

where $x$ is the input, $y$ is the target, and $\theta$ are model parameters.

## Point Prediction

A point prediction can be derived from the distribution. For classification:

$$
\hat{y}
=
\arg\max_y p_\theta(y\mid x)
$$

For regression with squared-error loss, the optimal point estimate is the conditional mean:

$$
\hat{y}
=
\mathbb{E}_{p_\theta(y\mid x)}[y]
$$

For absolute-error loss, the conditional median is often the corresponding decision.

## Predictive Distribution vs Score

Not every score is a probability. A model can output:

- Logits: unnormalized scores before [[concepts/architectures/softmax|softmax]] or sigmoid.
- Probabilities: normalized values that sum to one for mutually exclusive classes.
- Scores: arbitrary ranking values that may not be calibrated.
- Uncertainty estimates: variance, entropy, ensemble disagreement, or predictive intervals.

The evaluation metric should match the output type. Probability quality needs [[concepts/evaluation/probability-metrics|probability metrics]] and [[concepts/evaluation/calibration|calibration]], not only accuracy.

## Training

For a conditional likelihood model, training often minimizes negative log-likelihood:

$$
\mathcal{L}_{\mathrm{NLL}}(\theta)
=
-
\frac{1}{n}
\sum_{i=1}^{n}
\log p_\theta(y_i\mid x_i)
$$

This trains the full predictive distribution, while a downstream [[concepts/machine-learning/decision-rule|decision rule]] may later turn the distribution into a class, ranking, thresholded action, or abstention decision.

## Checks

- Is the output a calibrated probability, a score, a logit, or a hard label?
- Does the loss train the quantity needed by the downstream decision?
- Is the probability evaluated on held-out data under a fixed protocol?
- Are probability metrics separated from thresholded classification metrics?
- Is uncertainty used for a concrete decision such as abstention, triage, or active learning?

## Related

- [[concepts/machine-learning/decision-rule|Decision rule]]
- [[concepts/machine-learning/classification|Classification]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/machine-learning/cross-entropy-loss|Cross-entropy loss]]
- [[concepts/evaluation/probability-metrics|Probability metrics]]
- [[concepts/evaluation/proper-scoring-rule|Proper scoring rule]]
- [[concepts/evaluation/brier-score|Brier score]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/conformal-prediction|Conformal prediction]]
- [[concepts/evaluation/threshold-selection|Threshold selection]]
- [[concepts/architectures/softmax|Softmax]]
