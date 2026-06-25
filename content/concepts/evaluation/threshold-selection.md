---
title: Threshold Selection
tags:
  - evaluation
  - classification
  - decision
---

# Threshold Selection

Threshold selection converts a continuous score or probability into a discrete decision. It is separate from model training and must not be tuned on the test set.

For a score $s(x)$ and threshold $\tau$:

$$
\hat{y}
=
\mathbf{1}[s(x) \ge \tau]
$$

Changing $\tau$ changes the [[concepts/evaluation/confusion-matrix|confusion matrix]], precision, recall, false positive rate, and false negative rate.

## Choosing a Threshold

A threshold should be chosen on validation data according to a decision objective:

$$
\tau^\star
=
\arg\max_{\tau}
U(\mathrm{TP}(\tau), \mathrm{FP}(\tau), \mathrm{TN}(\tau), \mathrm{FN}(\tau))
$$

$U$ is a utility function or metric that encodes the cost of each decision outcome.

## Common Choices

- Maximize F1 when precision and recall are both important.
- Set recall above a required minimum and optimize precision.
- Set false positive rate below a deployment constraint.
- Choose a calibrated probability threshold when probability means decision risk.
- Use a domain-specific cost function when errors have asymmetric cost.

## Checks

- Was the threshold chosen before final test evaluation?
- Does the threshold match deployment cost rather than leaderboard convenience?
- Are probabilities calibrated enough for probability thresholds?
- Is the metric reported at a fixed threshold or across all thresholds?
- Is class prevalence in validation similar to deployment?

## Related

- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/confusion-matrix|Confusion matrix]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
