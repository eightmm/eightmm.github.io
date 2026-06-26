---
title: Reliability Diagram
tags:
  - evaluation
  - calibration
  - uncertainty
---

# Reliability Diagram

A reliability diagram visualizes calibration by comparing predicted confidence with empirical accuracy in bins.

Partition predictions into confidence bins $B_1,\ldots,B_K$. For bin $B_k$:

$$
\operatorname{conf}(B_k)
=
\frac{1}{|B_k|}
\sum_{i \in B_k}
\hat{p}_i
$$

$$
\operatorname{acc}(B_k)
=
\frac{1}{|B_k|}
\sum_{i \in B_k}
\mathbf{1}[\hat{y}_i = y_i]
$$

A calibrated model has:

$$
\operatorname{acc}(B_k)
\approx
\operatorname{conf}(B_k)
$$

for every bin with enough examples.

## Expected Calibration Error

Expected calibration error summarizes the bin gaps:

$$
\operatorname{ECE}
=
\sum_{k=1}^{K}
\frac{|B_k|}{n}
\left|
\operatorname{acc}(B_k)
-
\operatorname{conf}(B_k)
\right|
$$

## Confidence Definition

For multiclass classification, confidence is often:

$$
\hat{p}_i = \max_k p_\theta(y=k\mid x_i)
$$

and correctness is:

$$
\mathbf{1}[\arg\max_k p_\theta(y=k\mid x_i)=y_i]
$$

For binary classification, confidence can mean $p_\theta(y=1\mid x)$ or $\max(p_\theta(y=1\mid x), 1-p_\theta(y=1\mid x))$. State which one is used.

For regression, ranking, generation, or structured prediction, a reliability diagram needs a task-specific confidence event such as interval coverage, selective accuracy, or validity probability.

## Bin Design

| Choice | Risk |
| --- | --- |
| equal-width bins | high-confidence bins may have few examples |
| equal-count bins | bin boundaries differ across models |
| too many bins | noisy calibration estimates |
| too few bins | hides local miscalibration |
| test-set tuning | overstates calibration quality |

Calibration estimates should include uncertainty when the bin counts are small:

$$
\widehat{\operatorname{acc}}(B_k)
\pm
1.96
\sqrt{
\frac{\widehat{p}_k(1-\widehat{p}_k)}
{|B_k|}
}
$$

where $\widehat{p}_k=\operatorname{acc}(B_k)$.

## Reading the Plot

| Pattern | Interpretation |
| --- | --- |
| accuracy below confidence | overconfident predictions |
| accuracy above confidence | underconfident predictions |
| low-confidence bins sparse | uncertainty claim is weak in that range |
| high-confidence gap | risky for automated decision or filtering |

Calibration does not replace accuracy. A model can be well-calibrated and still wrong often, or accurate but overconfident.

## Checks

- Are bins populated enough to be meaningful?
- Is calibration measured on held-out data?
- Is confidence defined consistently for binary, multiclass, or structured output?
- Does calibration hold under distribution shift?
- Are accuracy and calibration both reported?
- Are binning choices fixed before comparing models?

## Related

- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/probability-metrics|Probability metrics]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/bootstrap-evaluation|Bootstrap evaluation]]
