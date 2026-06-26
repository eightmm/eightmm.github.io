---
title: Classification Metrics
tags:
  - evaluation
  - classification
  - metrics
---

# Classification Metrics

Classification metrics evaluate discrete label predictions. The right metric depends on [[concepts/data/class-imbalance|class balance]], error cost, thresholding, and whether the model is used as a hard classifier, ranker, or probability estimator.

For binary classification, define:

- TP: true positives
- FP: false positives
- TN: true negatives
- FN: false negatives

Equivalently:

| True label | Predicted positive | Predicted negative |
|---|---:|---:|
| Positive | TP | FN |
| Negative | FP | TN |

Most classification metrics are functions of this [[concepts/evaluation/confusion-matrix|confusion matrix]]:

$$
M(\tau)
=
f(\mathrm{TP}(\tau),\mathrm{FP}(\tau),\mathrm{TN}(\tau),\mathrm{FN}(\tau))
$$

The explicit $\tau$ matters because a probability or score becomes a class only after [[concepts/evaluation/threshold-selection|threshold selection]].

## Metric Families

| Family | Question | Examples |
|---|---|---|
| Hard-label metrics | Are final decisions correct? | accuracy, balanced accuracy, F1 |
| Threshold-swept metrics | Is ranking useful across thresholds? | AUROC, AUPRC |
| Probability metrics | Are probabilities honest? | NLL, Brier score |
| Operational metrics | Does the model meet a use constraint? | recall at fixed FPR, precision at fixed recall |

Do not report only one family unless it matches the actual claim. A screening model, a clinical triage classifier, and a benchmark classifier need different evidence.

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

F1 is threshold-dependent. It is useful when precision and recall both matter, but it hides which side failed.

The $F_\beta$ score makes the precision/recall tradeoff explicit:

$$
F_\beta
=
(1+\beta^2)
\frac{
\operatorname{Precision}\cdot\operatorname{Recall}
}{
\beta^2\operatorname{Precision}+\operatorname{Recall}
}
$$

Use $\beta>1$ when recall is more important, and $\beta<1$ when precision is more important.

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

## Specificity and False Positive Rate

Specificity measures how many negatives are rejected:

$$
\operatorname{Specificity}
=
\frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}}
$$

False positive rate is:

$$
\operatorname{FPR}
=
\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}
=
1-\operatorname{Specificity}
$$

In rare-positive settings, a small FPR can still produce many false positives when the negative pool is large.

## Matthews Correlation Coefficient

Matthews correlation coefficient summarizes all four confusion-matrix cells:

$$
\operatorname{MCC}
=
\frac{
\mathrm{TP}\mathrm{TN}-\mathrm{FP}\mathrm{FN}
}{
\sqrt{
(\mathrm{TP}+\mathrm{FP})
(\mathrm{TP}+\mathrm{FN})
(\mathrm{TN}+\mathrm{FP})
(\mathrm{TN}+\mathrm{FN})
}
}
$$

MCC is often more informative than accuracy when classes are imbalanced, because it penalizes degenerate classifiers that predict only one class.

## AUROC and AUPRC

AUROC measures whether positives tend to receive higher scores than negatives:

$$
\operatorname{AUROC}
=
\Pr(s(x^+) > s(x^-))
$$

This is a ranking statement, not a calibrated probability statement.

AUPRC is usually more informative when positives are rare. The precision-recall curve is:

$$
\left(
\operatorname{Recall}(\tau),
\operatorname{Precision}(\tau)
\right)
$$

as $\tau$ varies. See [[concepts/evaluation/precision-recall|Precision recall]].

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

## Multi-Class and Multi-Label Cases

For multi-class classification with one true label, compute macro, micro, and weighted averages carefully:

| Average | Meaning | Risk |
|---|---|---|
| Macro | Unweighted average over classes | highlights rare-class failures |
| Micro | Counts all examples globally | dominated by frequent classes |
| Weighted | Class-average weighted by support | can hide minority collapse |

For multi-label classification, each label has its own binary decision:

$$
\hat{y}_{ik}
=
\mathbf{1}[p_{ik}\ge \tau_k]
$$

where the threshold $\tau_k$ may need to differ by label. Reporting only global accuracy is usually too weak.

## Bio-AI Examples

- Activity classification: define active/inactive labels from assay values before reporting metrics.
- Virtual screening: AUPRC, enrichment, and recall at fixed false-positive budget often matter more than raw accuracy.
- Toxicity prediction: false negatives and false positives can have asymmetric cost.
- Protein function labels: multi-label metrics are often required because one protein can have multiple annotations.

When labels come from heterogeneous assays, metric interpretation depends on [[concepts/evaluation/assay-harmonization|Assay harmonization]] and label semantics, not only model predictions.

## Checks

- Is class imbalance severe?
- Is the decision threshold fixed, tuned, or deployment-specific?
- Are false positives and false negatives equally costly?
- Are probabilities calibrated or only class labels needed?
- Should the report include AUROC, AUPRC, or ranking-style metrics?
- Are macro, micro, and per-class metrics reported for imbalanced multi-class tasks?
- Are labels derived from noisy, censored, or assay-dependent measurements?

## Related

- [[concepts/machine-learning/classification|Classification]]
- [[concepts/data/class-imbalance|Class imbalance]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/confusion-matrix|Confusion matrix]]
- [[concepts/evaluation/threshold-selection|Threshold selection]]
- [[concepts/evaluation/probability-metrics|Probability metrics]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/precision-recall|Precision recall]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
