---
title: Classification Metrics
tags:
  - evaluation
  - classification
  - metrics
---

# Classification Metrics

Classification metrics evaluate discrete label predictions. The right metric depends on class balance, error cost, thresholding, and whether probabilities or only class decisions matter.

For binary classification, define:

- TP: true positives
- FP: false positives
- TN: true negatives
- FN: false negatives

## Accuracy

Accuracy measures the fraction of correct predictions:

$$
\operatorname{Accuracy}
=
\frac{\mathrm{TP}+\mathrm{TN}}
{\mathrm{TP}+\mathrm{FP}+\mathrm{TN}+\mathrm{FN}}
$$

It is easy to read but can be misleading under severe class imbalance.

## Precision, Recall, and F1

Precision asks how many predicted positives are correct:

$$
\operatorname{Precision}
=
\frac{\mathrm{TP}}
{\mathrm{TP}+\mathrm{FP}}
$$

Recall asks how many true positives are recovered:

$$
\operatorname{Recall}
=
\frac{\mathrm{TP}}
{\mathrm{TP}+\mathrm{FN}}
$$

F1 is their harmonic mean:

$$
F_1
=
2
\frac{
\operatorname{Precision}\cdot\operatorname{Recall}
}{
\operatorname{Precision}+\operatorname{Recall}
}
$$

## Balanced Accuracy

Balanced accuracy averages sensitivity and specificity:

$$
\operatorname{BalancedAccuracy}
=
\frac{1}{2}
\left(
\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}
+
\frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}}
\right)
$$

It is useful when positive and negative classes are imbalanced.

## Probability Metrics

When predicted probabilities matter, use [[concepts/evaluation/probability-metrics|probability-aware metrics]] such as negative log-likelihood or Brier score:

$$
\operatorname{NLL}
=
-\frac{1}{n}
\sum_{i=1}^{n}
\log p_\theta(y_i\mid x_i)
$$

$$
\operatorname{Brier}
=
\frac{1}{n}
\sum_{i=1}^{n}
(p_i-y_i)^2
$$

where $p_i$ is the predicted probability for the positive class.

## Checks

- Is class imbalance severe?
- Is the decision threshold fixed, tuned, or deployment-specific?
- Are false positives and false negatives equally costly?
- Are probabilities calibrated or only class labels needed?
- Should the report include AUROC, AUPRC, or ranking-style metrics?

## Related

- [[concepts/machine-learning/classification|Classification]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/confusion-matrix|Confusion matrix]]
- [[concepts/evaluation/threshold-selection|Threshold selection]]
- [[concepts/evaluation/probability-metrics|Probability metrics]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
