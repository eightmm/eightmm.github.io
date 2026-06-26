---
title: Metric
tags:
  - evaluation
  - methodology
---

# Metric

A metric is the number used to summarize model behavior under a specific evaluation protocol. A good metric matches the decision that will be made from the model, not just the training loss.

For a test set $\mathcal{D}_{\mathrm{test}}=\{(x_j,y_j)\}_{j=1}^{m}$, a generic metric is:

$$
\hat{M}(f)
= \frac{1}{m}\sum_{j=1}^{m}
m(f(x_j), y_j)
$$

where $m(\cdot)$ is the per-example score.

Classification accuracy is:

$$
\operatorname{Acc}
= \frac{1}{m}\sum_{j=1}^{m}
\mathbf{1}\left[\arg\max_k p_\theta(k\mid x_j)=y_j\right]
$$

Mean absolute error is:

$$
\operatorname{MAE}
= \frac{1}{m}\sum_{j=1}^{m}
\lvert \hat{y}_j-y_j\rvert
$$

## Metric as an Estimator

A reported metric is an estimator of behavior under a target distribution:

$$
M(f)
=
\mathbb{E}_{(x,y)\sim p_{\mathrm{target}}}
\left[
m(f(x),y)
\right]
$$

The finite test-set estimate is:

$$
\hat{M}(f)
=
\frac{1}{m}
\sum_{j=1}^{m}
m(f(x_j),y_j)
$$

This is why split design and confidence intervals matter. A metric can be precisely computed on the wrong test set and still support the wrong claim.

## Primary and Diagnostic Metrics

The primary metric decides model selection or headline comparison:

$$
f^\*
=
\arg\max_{f\in\mathcal{F}}
M_{\mathrm{primary}}(f;\mathcal{D}_{\mathrm{val}})
$$

Diagnostic metrics explain failure modes:

$$
\{M_{\mathrm{diagnostic},k}\}_{k=1}^{K}
$$

Examples include calibration, subgroup error, invalid-output rate, uncertainty coverage, latency, cost, or robustness slices. Diagnostics should not silently become selection criteria after seeing results.

## Aggregation

Metric aggregation should match the example unit. Per-row averaging can overweight duplicated molecules, long documents, large protein families, or repeated prompts. Grouped aggregation can be written as:

$$
\hat{M}_{\mathrm{group}}
=
\frac{1}{G}
\sum_{g=1}^{G}
\left[
\frac{1}{|D_g|}
\sum_{(x,y)\in D_g}
m(f(x),y)
\right]
$$

where $D_g$ is a group such as a target, scaffold, query, protein family, dataset source, or user.

## Checks

- Does the metric match the scientific or product decision?
- Has the metric been selected using [[concepts/evaluation/metric-selection|Metric selection]] rather than habit?
- Is higher or lower better, and is the scale interpretable?
- Does class imbalance require AUROC, AUPRC, balanced accuracy, or calibration?
- Are hard-decision, ranking, probability, and calibration metrics being mixed?
- Are confidence intervals or multiple seeds needed?
- Is the metric valid under the chosen split?
- Is the metric averaged per row, per group, per query, per target, or per task?
- Are invalid, abstained, timed-out, or failed outputs counted rather than silently dropped?
- Is the primary metric fixed before model selection?

## Related

- [[concepts/evaluation/index|Evaluation]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/regression-metrics|Regression metrics]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/probability-metrics|Probability metrics]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
- [[concepts/evaluation/failure-mode-taxonomy|Failure mode taxonomy]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/machine-learning/loss-function|Loss function]]
