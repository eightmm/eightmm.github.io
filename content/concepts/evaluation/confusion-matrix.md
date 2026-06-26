---
title: Confusion Matrix
tags:
  - evaluation
  - classification
  - metrics
---

# Confusion Matrix

A confusion matrix counts how predicted classes differ from true classes. It is the base table behind many classification metrics.

For binary classification:

|  | Predicted positive | Predicted negative |
| --- | --- | --- |
| True positive class | TP | FN |
| True negative class | FP | TN |

The total number of evaluated examples is:

$$
n = \mathrm{TP}+\mathrm{FP}+\mathrm{TN}+\mathrm{FN}
$$

## Derived Quantities

True positive rate, or recall:

$$
\operatorname{TPR}
=
\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}
$$

False positive rate:

$$
\operatorname{FPR}
=
\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}
$$

Positive predictive value, or precision:

$$
\operatorname{PPV}
=
\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}
$$

True negative rate, or specificity:

$$
\operatorname{TNR}
=
\frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}}
$$

False negative rate:

$$
\operatorname{FNR}
=
\frac{\mathrm{FN}}{\mathrm{TP}+\mathrm{FN}}
$$

Accuracy:

$$
\operatorname{Accuracy}
=
\frac{\mathrm{TP}+\mathrm{TN}}{n}
$$

Balanced accuracy:

$$
\operatorname{BalancedAccuracy}
=
\frac{1}{2}
(\operatorname{TPR}+\operatorname{TNR})
$$

Balanced accuracy is often more informative than raw accuracy under class imbalance.

## Why It Matters

Aggregate metrics hide which error type dominates. In imbalanced problems, a model can have high accuracy while missing most positives or producing too many false positives.

## Threshold and Prevalence

The confusion matrix is defined after a decision rule. For a score $s(x)$:

$$
\hat{y}=\mathbf{1}[s(x)\ge\tau]
$$

Changing $\tau$ changes TP, FP, TN, and FN. A test-set confusion matrix should use a threshold chosen from validation data or a fixed application rule.

Prevalence changes how errors feel operationally. With rare positives, even a low FPR can produce many false positives:

$$
\mathrm{FP}
\approx
(1-\pi)N\operatorname{FPR}
$$

where $\pi$ is the positive prevalence and $N$ is the evaluation size.

## Multi-Class Form

For $K$ classes, the confusion matrix is:

$$
C_{ab}
=
\#\{i: y_i=a,\ \hat{y}_i=b\}
$$

Rows often represent true classes and columns predicted classes. State the convention because transposed matrices invert the reading of common errors.

## Checks

- Which class is treated as positive?
- Are classes imbalanced?
- Is the threshold fixed before test evaluation?
- Are false positives and false negatives equally costly?
- Should metrics be reported per class, macro-averaged, or micro-averaged?
- Does the paper show the confusion matrix or only a single derived metric?

## Related

- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/precision-recall|Precision and recall]]
- [[concepts/evaluation/threshold-selection|Threshold selection]]
- [[concepts/evaluation/error-analysis|Error analysis]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/machine-learning/classification|Classification]]
