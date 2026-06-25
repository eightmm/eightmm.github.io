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

## Checks

- Are bins populated enough to be meaningful?
- Is calibration measured on held-out data?
- Is confidence defined consistently for binary, multiclass, or structured output?
- Does calibration hold under distribution shift?
- Are accuracy and calibration both reported?

## Related

- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/bootstrap-evaluation|Bootstrap evaluation]]
