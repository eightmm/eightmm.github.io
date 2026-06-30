---
title: Classification
tags:
  - machine-learning
  - supervised-learning
---

# Classification

Classification predicts a discrete class label. The model usually outputs a [[concepts/machine-learning/probabilistic-prediction|probability distribution]] over classes, then a [[concepts/machine-learning/decision-rule|decision rule]] converts that distribution into a class.

For $K$ classes, logits $z\in\mathbb{R}^{K}$ become probabilities through softmax:

$$
p_\theta(y=k\mid x)
= \frac{\exp(z_k)}
{\sum_{\ell=1}^{K}\exp(z_\ell)}
$$

The predicted class is:

$$
\hat{y}
= \arg\max_k p_\theta(y=k\mid x)
$$

The standard supervised loss is [[concepts/machine-learning/cross-entropy-loss|cross-entropy]]:

$$
\mathcal{L}_{\mathrm{CE}}
= -\sum_{k=1}^{K} y_k\log p_\theta(y=k\mid x)
$$

For binary classification, this becomes:

$$
\mathcal{L}_{\mathrm{BCE}}
=
-y\log p_\theta(y=1\mid x)
-(1-y)\log(1-p_\theta(y=1\mid x))
$$

For a binary logit $z$, the probability is:

$$
p_\theta(y=1\mid x)
=
\sigma(z)
=
\frac{1}{1+\exp(-z)}
$$

The logit is not a probability. It is an unbounded score that becomes a probability only after the sigmoid or softmax link function.

## Empirical Risk

For a labeled dataset $\{(x_i,y_i)\}_{i=1}^{n}$, classification training often minimizes:

$$
\hat{R}(\theta)
=
\frac{1}{n}\sum_{i=1}^{n}
\mathcal{L}_{\mathrm{CE}}
(f_\theta(x_i), y_i)
$$

If classes are imbalanced, a weighted objective may be used:

$$
\hat{R}_{w}(\theta)
=
\frac{1}{n}\sum_{i=1}^{n}
w_{y_i}\,
\mathcal{L}_{\mathrm{CE}}
(f_\theta(x_i), y_i)
$$

The weights $w_k$ change the training objective. They do not automatically fix the decision threshold or evaluation metric.

## Decision Rule

Probabilities and decisions are different. A threshold $\tau$ converts a probability into a label:

$$
\hat{y}
=
\mathbb{1}\left[p_\theta(y=1\mid x)\ge \tau\right]
$$

Changing $\tau$ changes precision, recall, false positives, and false negatives without changing the model itself.

For binary classification:

| Predicted / True | $y=1$ | $y=0$ |
| --- | --- | --- |
| $\hat{y}=1$ | true positive | false positive |
| $\hat{y}=0$ | false negative | true negative |

Common decision metrics include:

$$
\mathrm{precision}
=
\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}},
\qquad
\mathrm{recall}
=
\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}
$$

The threshold should be selected on validation data and reported with the metric.

## Multi-Label Case

If labels are not mutually exclusive, each class often uses an independent sigmoid:

$$
p_k = \sigma(z_k),
\qquad
\hat{y}_k=\mathbb{1}[p_k\ge \tau_k]
$$

This is different from softmax classification, where probabilities compete across classes.

## Calibration Boundary

A classifier can rank examples correctly while reporting overconfident probabilities. Calibration asks whether predicted probabilities match observed frequencies:

$$
P(y=1\mid \hat{p}=p) \approx p
$$

Calibration matters when probabilities drive decisions, risk estimates, uncertainty filtering, or downstream expected utility. If only ranking matters, ranking metrics may be more relevant than probability calibration.

## Task Variants

| Variant | Output | Caution |
| --- | --- | --- |
| binary | one probability or logit | threshold controls decision behavior |
| multiclass | one of $K$ mutually exclusive classes | softmax assumes competition |
| multi-label | independent label vector | per-class thresholds and label correlations matter |
| ordinal | ordered categories | treating labels as unordered may waste structure |
| hierarchical | taxonomy path or nested labels | parent-child consistency may be required |

## Checks

- Are labels mutually exclusive, multi-label, ordinal, or hierarchical?
- Is class imbalance severe enough to require weighting or threshold tuning?
- Does train/validation/test prevalence match the deployment decision?
- Are probabilities calibrated, or only rankings/classes needed?
- Does the split prevent near-duplicate examples from crossing train and test?
- Is the selected threshold fixed before test evaluation?
- Does the reported metric reflect the cost of false positives and false negatives?
- Are logits, probabilities, rankings, and final labels kept distinct?

## Related

- [[concepts/machine-learning/empirical-risk-minimization|Empirical risk minimization]]
- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/probabilistic-prediction|Probabilistic prediction]]
- [[concepts/machine-learning/decision-rule|Decision rule]]
- [[concepts/machine-learning/cross-entropy-loss|Cross-entropy loss]]
- [[concepts/data/class-imbalance|Class imbalance]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/confusion-matrix|Confusion matrix]]
- [[concepts/evaluation/precision-recall|Precision and recall]]
- [[concepts/evaluation/threshold-selection|Threshold selection]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/proper-scoring-rule|Proper scoring rule]]
- [[concepts/architectures/softmax|Softmax]]
