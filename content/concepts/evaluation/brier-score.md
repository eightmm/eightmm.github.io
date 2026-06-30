---
title: Brier Score
tags:
  - evaluation
  - probability
  - calibration
---

# Brier Score

Brier score measures the squared error of predicted probabilities. It is a [[concepts/evaluation/proper-scoring-rule|Proper scoring rule]], so it rewards honest probability estimates rather than only hard decisions.

For binary labels $y_i\in\{0,1\}$ and predicted probability $p_i=p_\theta(y=1\mid x_i)$:

$$
\operatorname{Brier}
=
\frac{1}{n}
\sum_{i=1}^{n}
(p_i-y_i)^2
$$

For multiclass prediction with one-hot target $y_{ik}$:

$$
\operatorname{Brier}
=
\frac{1}{n}
\sum_{i=1}^{n}
\sum_{k=1}^{K}
(p_{ik}-y_{ik})^2
$$

Lower is better.

For a deterministic hard classifier that predicts $p_i\in\{0,1\}$, binary Brier score becomes the error rate. For probabilistic models, it also rewards being less overconfident when uncertain.

## Interpretation

Brier score penalizes both:

- wrong class decisions
- poorly calibrated probabilities

An overconfident wrong prediction receives a large penalty. A correct but underconfident prediction is also penalized, though less severely.

For binary prediction, the per-example score is bounded:

$$
0 \le (p-y)^2 \le 1
$$

This bounded scale can make Brier score easier to communicate than NLL, but it may be less sensitive to extremely overconfident mistakes.

## Decomposition

For binary forecasts grouped into probability bins, Brier score can be decomposed into calibration and refinement terms:

$$
\operatorname{BS}
=
\text{reliability}
-
\text{resolution}
+
\text{uncertainty}
$$

Informally:

| Term | Meaning |
| --- | --- |
| reliability | predicted probabilities match observed frequencies |
| resolution | predictions separate high-risk and low-risk cases |
| uncertainty | base-rate difficulty of the event |

This is why two models with similar Brier scores can behave differently: one may be well-calibrated but uninformative, while another may separate cases well but need calibration.

## Comparison With NLL

| Metric | Rewards | Risk |
| --- | --- | --- |
| Brier score | squared probability accuracy | less sensitive to extreme overconfidence |
| negative log-likelihood | probability assigned to true label | unbounded penalty for confident errors |
| accuracy | hard decision correctness | ignores probability quality |
| expected calibration error | calibration bins | depends on binning and ignores sharpness |

If probabilities drive decisions, use Brier or NLL with calibration plots. If only hard decisions matter, also report thresholded metrics.

## Relation to Calibration

Brier score summarizes probability quality as one scalar. [[concepts/evaluation/calibration|Calibration]] asks a different question: do examples predicted near probability $p$ occur with frequency $p$?

Use both:

$$
\text{Brier score}
\quad
\text{and}
\quad
\text{reliability diagram}
$$

when confidence is part of the decision.

## Checks

- Are probabilities calibrated on validation data, not test data?
- Is Brier score reported on the same split and unit as other metrics?
- Are class imbalance and prevalence changes considered?
- Is the model evaluated with both probability metrics and thresholded decision metrics?
- Is the score used only when probabilities are meaningful?
- Are probability outputs clipped, rounded, or post-hoc calibrated before scoring?
- Is the event definition consistent across train, validation, and test?

## Related

- [[concepts/evaluation/probability-metrics|Probability metrics]]
- [[concepts/evaluation/proper-scoring-rule|Proper scoring rule]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/reliability-diagram|Reliability diagram]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/machine-learning/probabilistic-prediction|Probabilistic prediction]]
