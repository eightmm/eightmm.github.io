---
title: Threshold Selection
tags:
  - evaluation
  - classification
  - decision
---

# Threshold Selection

Threshold selection converts a continuous score or probability into a discrete [[concepts/machine-learning/decision-rule|decision]]. It is separate from model training and must not be tuned on the test set.

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

When probabilities are calibrated and costs are known, a binary action can be chosen by expected cost. Predict positive when:

$$
p(y=1\mid x) C_{\mathrm{FN}}
\ge
(1-p(y=1\mid x)) C_{\mathrm{FP}}
$$

This gives the threshold:

$$
\tau
=
\frac{C_{\mathrm{FP}}}
{C_{\mathrm{FP}}+C_{\mathrm{FN}}}
$$

where $C_{\mathrm{FP}}$ is the cost of a false positive and $C_{\mathrm{FN}}$ is the cost of a false negative. This only works when the score is a calibrated probability and the costs match deployment.

## Common Choices

- Maximize F1 when precision and recall are both important.
- Set recall above a required minimum and optimize precision.
- Set false positive rate below a deployment constraint.
- Choose a calibrated probability threshold when probability means decision risk.
- Choose an abstention threshold when low-confidence predictions should be deferred.
- Use a domain-specific cost function when errors have asymmetric cost.

## Threshold Report

| Field | Why |
| --- | --- |
| score definition | logits, probability, distance, anomaly score, or rank score are not interchangeable |
| selection split | threshold must be chosen on validation or calibration data |
| prevalence | precision and alert load depend on base rate |
| utility or constraint | explains why this threshold is useful |
| frozen threshold | prevents test-set tuning |
| per-slice behavior | rare classes or subgroups can fail at the same threshold |

## Checks

- Was the threshold chosen before final test evaluation?
- Does the threshold match deployment cost rather than leaderboard convenience?
- Are probabilities calibrated enough for probability thresholds?
- Is the metric reported at a fixed threshold or across all thresholds?
- If abstention is allowed, are coverage and selective risk reported?
- Is class prevalence in validation similar to deployment?
- Is the threshold robust across seeds, time, source, or domain slices?

## Related

- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/machine-learning/probabilistic-prediction|Probabilistic prediction]]
- [[concepts/machine-learning/decision-rule|Decision rule]]
- [[concepts/evaluation/confusion-matrix|Confusion matrix]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/selective-prediction|Selective prediction]]
- [[concepts/evaluation/precision-recall|Precision and recall]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
