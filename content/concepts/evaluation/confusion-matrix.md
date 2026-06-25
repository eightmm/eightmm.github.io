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

## Why It Matters

Aggregate metrics hide which error type dominates. In imbalanced problems, a model can have high accuracy while missing most positives or producing too many false positives.

## Checks

- Which class is treated as positive?
- Are classes imbalanced?
- Is the threshold fixed before test evaluation?
- Are false positives and false negatives equally costly?
- Should metrics be reported per class, macro-averaged, or micro-averaged?

## Related

- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/threshold-selection|Threshold selection]]
- [[concepts/evaluation/error-analysis|Error analysis]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/machine-learning/classification|Classification]]
