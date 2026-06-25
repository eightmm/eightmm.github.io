---
title: Decision Rule
tags:
  - machine-learning
  - evaluation
  - decision
---

# Decision Rule

A decision rule converts a model output into an action: class label, thresholded alert, ranking, abstention, generated candidate, or human-review route. It should be defined by the downstream cost or utility, not only by model convenience.

Given a predictive distribution $p_\theta(y\mid x)$ and action set $\mathcal{A}$, the risk of action $a$ is:

$$
\mathcal{R}(a\mid x)
=
\mathbb{E}_{y\sim p_\theta(y\mid x)}
\left[
C(a,y)
\right]
$$

where $C(a,y)$ is the cost of taking action $a$ when the true outcome is $y$.

The Bayes decision minimizes expected cost:

$$
a^\*(x)
=
\arg\min_{a\in\mathcal{A}}
\mathcal{R}(a\mid x)
$$

## Classification

With zero-one loss:

$$
C(a,y)=\mathbb{1}[a\ne y]
$$

the optimal decision is the maximum-probability class:

$$
\hat{y}
=
\arg\max_y p_\theta(y\mid x)
$$

With asymmetric costs, the best threshold can move away from $0.5$.

## Binary Threshold

For binary prediction with probability $p=p_\theta(y=1\mid x)$, choose positive action if:

$$
p \cdot C(\mathrm{positive},1)
+
(1-p) \cdot C(\mathrm{positive},0)
\le
p \cdot C(\mathrm{negative},1)
+
(1-p) \cdot C(\mathrm{negative},0)
$$

In practice, this becomes a threshold rule:

$$
\hat{y}
=
\mathbb{1}[p \ge \tau]
$$

where $\tau$ should be selected on validation data according to the decision objective.

## Abstention

If the model can defer to another process, abstention is another action:

$$
\mathcal{A}
=
\{\text{class }1,\ldots,\text{class }K,\text{abstain}\}
$$

Abstention is useful only if its cost is lower than the expected cost of a likely wrong prediction.

## Checks

- What action is taken from the model output?
- What cost or utility defines a good action?
- Is the threshold or policy selected on validation data, not test data?
- Are probabilities calibrated enough for cost-sensitive decisions?
- Are hard-decision metrics separated from probability and ranking metrics?
- Does the decision rule remain valid under dataset shift?

## Related

- [[concepts/machine-learning/probabilistic-prediction|Probabilistic prediction]]
- [[concepts/machine-learning/classification|Classification]]
- [[concepts/evaluation/threshold-selection|Threshold selection]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/probability-metrics|Probability metrics]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
