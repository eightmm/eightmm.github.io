---
title: Precision and Recall
tags:
  - evaluation
  - metrics
  - classification
---

# Precision and Recall

Precision and recall describe different failure costs for positive predictions.

For binary decisions:

$$
\operatorname{Precision}
=
\frac{\mathrm{TP}}
{\mathrm{TP}+\mathrm{FP}}
$$

$$
\operatorname{Recall}
=
\frac{\mathrm{TP}}
{\mathrm{TP}+\mathrm{FN}}
$$

Precision asks whether predicted positives are trustworthy. Recall asks whether true positives are recovered.

## Thresholded Decisions

If a model produces a score $s(x)$, a threshold $\tau$ creates a decision:

$$
\hat{y}
=
\mathbf{1}[s(x)\ge\tau]
$$

Changing $\tau$ changes precision and recall:

$$
\tau
\uparrow
\Rightarrow
\operatorname{Precision}
\text{ often increases},
\quad
\operatorname{Recall}
\text{ often decreases}
$$

The threshold should be selected on validation data, not on the final test set.

## Retrieval and Detection

Precision and recall also apply beyond classification:

- Retrieval: precision asks how many returned items are relevant; recall asks how many relevant items were found.
- Object detection: precision and recall depend on class, confidence threshold, and IoU matching.
- Virtual screening: early precision or enrichment may matter more than global recall.

## Prevalence Boundary

Precision depends on class prevalence:

$$
\operatorname{Precision}
=
\frac{\pi\operatorname{TPR}}
{\pi\operatorname{TPR} + (1-\pi)\operatorname{FPR}}
$$

where $\pi=P(y=1)$ is prevalence. The same classifier can have very different precision when deployment prevalence differs from the benchmark.

Recall does not depend on the number of true negatives, but it depends on what was labeled as positive and whether positives are complete.

## F Score

The $F_\beta$ score combines precision and recall:

$$
F_\beta
=
(1+\beta^2)
\frac{\operatorname{Precision}\cdot\operatorname{Recall}}
{\beta^2\operatorname{Precision}+\operatorname{Recall}}
$$

$\beta>1$ emphasizes recall, while $\beta<1$ emphasizes precision. Report $\beta$ and the threshold-selection rule.

## Averaging

For multi-class or multi-label tasks:

| Averaging | Meaning |
| --- | --- |
| micro | pool TP/FP/FN globally |
| macro | average class-level metrics equally |
| weighted | average class-level metrics by support |
| per-class | report each label separately |

Macro metrics reveal rare-class failures. Micro metrics can hide them.

## Checks

- What counts as a positive?
- Are false positives or false negatives more costly?
- Is the threshold fixed before final evaluation?
- Is prevalence similar between validation, test, and deployment?
- Is precision/recall reported per class, macro-averaged, or micro-averaged?
- Are labels complete enough that missing positives are not counted as false positives?

## Related

- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/confusion-matrix|Confusion matrix]]
- [[concepts/evaluation/threshold-selection|Threshold selection]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/tasks/object-detection|Object detection]]
