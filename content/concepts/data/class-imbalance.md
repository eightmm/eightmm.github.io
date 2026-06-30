---
title: Class Imbalance
tags:
  - data
  - evaluation
  - classification
---

# Class Imbalance

Class imbalance occurs when target classes have very different frequencies. It changes what a metric means, how batches behave during training, and how decision thresholds should be selected.

For binary classification, the base rate is:

$$
\pi
=
P(y=1)
$$

In a dataset $\mathcal{D}$, the empirical base rate is:

$$
\hat{\pi}
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbb{1}[y_i=1]
$$

When $\hat{\pi}$ is very small, a model can achieve high accuracy by predicting the majority class while missing the examples that matter.

## Prevalence Shift

Training, validation, test, and deployment can have different base rates:

$$
\pi_{\mathrm{train}}
\ne
\pi_{\mathrm{deploy}}
$$

This matters because precision depends on prevalence:

$$
\operatorname{PPV}
=
\frac{\operatorname{TPR}\pi}
{\operatorname{TPR}\pi+\operatorname{FPR}(1-\pi)}
$$

Even if sensitivity and specificity stay fixed, deployment precision can drop sharply when positives are rare.

## Why It Matters

- Accuracy can be misleading when one class dominates.
- Mini-batches may contain too few rare positives for stable gradients.
- Thresholds that work in validation may fail when deployment prevalence changes.
- AUROC can look strong even when precision is poor at useful recall.
- Calibration depends on the target prevalence and sampling process.

## Common Responses

- Use metrics that reflect rare-class behavior: recall, precision, F1, AUPRC, balanced accuracy.
- Use stratified or balanced sampling during training, while evaluating on the target distribution.
- Use class weights or focal-style losses when false negatives or rare positives matter.
- Tune thresholds on validation data using a decision objective.
- Report prevalence in train, validation, test, and deployment-like slices.

Weighted binary cross-entropy is one simple training response:

$$
\mathcal{L}
=
-
w_1 y\log p_\theta(y=1\mid x)
-
w_0(1-y)\log(1-p_\theta(y=1\mid x))
$$

where $w_1$ and $w_0$ control the cost of positive and negative errors.

## Sampling vs Evaluation

Balanced training batches change the training distribution:

$$
q_{\mathrm{batch}}(y)
\ne
p_{\mathrm{eval}}(y)
$$

This can help optimization but can also distort probability calibration. If training uses balanced sampling, evaluate metrics and calibration on the target prevalence or recalibrate probabilities on a representative validation set.

## Metric Boundary

| Metric | Useful when | Caveat |
| --- | --- | --- |
| accuracy | classes are balanced and costs similar | hides rare-class failure |
| AUROC | ranking across thresholds | can look good with poor precision |
| AUPRC | positive class is rare | depends strongly on prevalence |
| balanced accuracy | both classes should matter equally | ignores probability quality |
| calibrated probability | decisions use risk thresholds | needs representative calibration data |

## Checks

- What is the class prevalence in each split?
- Is the validation prevalence similar to the deployment setting?
- Are metrics threshold-free, thresholded, or probability-based?
- Are positives rare because of true prevalence or collection bias?
- Are missing labels being treated as negatives?
- Does batch sampling differ from evaluation sampling?
- Are precision/recall reported at a useful operating point?
- Is probability calibration checked under deployment-like prevalence?

## Related

- [[concepts/data/sampling-strategy|Sampling strategy]]
- [[concepts/data/sampling-bias|Sampling bias]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/label-noise|Label noise]]
- [[concepts/machine-learning/classification|Classification]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/confusion-matrix|Confusion matrix]]
- [[concepts/evaluation/threshold-selection|Threshold selection]]
- [[concepts/evaluation/probability-metrics|Probability metrics]]
- [[concepts/evaluation/calibration|Calibration]]
