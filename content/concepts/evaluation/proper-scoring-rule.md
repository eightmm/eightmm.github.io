---
title: Proper Scoring Rule
tags:
  - evaluation
  - probability
  - metrics
---

# Proper Scoring Rule

A proper scoring rule evaluates probabilistic predictions in a way that rewards honest probabilities. If the true conditional distribution is $q(y\mid x)$, the best expected score should be achieved by predicting $q$, not by reporting an overconfident or distorted distribution.

For a loss-style score $S(p,y)$ where lower is better, propriety means:

$$
\mathbb{E}_{Y\sim q}
[S(q,Y)]
\le
\mathbb{E}_{Y\sim q}
[S(p,Y)]
$$

for any predicted distribution $p$.

If equality holds only when $p=q$, the rule is strictly proper:

$$
\mathbb{E}_{Y\sim q}
[S(q,Y)]
<
\mathbb{E}_{Y\sim q}
[S(p,Y)]
\quad
\text{for } p\ne q
$$

Strict propriety matters when the goal is not only ranking decisions, but learning or evaluating the full predictive distribution.

## Why It Matters

Accuracy only evaluates the final class decision. A proper scoring rule evaluates the full predictive distribution:

$$
p_\theta(y\mid x)
$$

This matters when probabilities are used for [[concepts/evaluation/threshold-selection|Threshold selection]], [[concepts/evaluation/calibration|Calibration]], [[concepts/evaluation/selective-prediction|Selective prediction]], active learning, or risk-sensitive decisions.

## Common Examples

Negative log-likelihood for a discrete target:

$$
S_{\mathrm{NLL}}(p,y)
=
-\log p(y)
$$

Brier score for a binary target:

$$
S_{\mathrm{Brier}}(p,y)
=
(p-y)^2
$$

Continuous ranked probability score for a real-valued target:

$$
\operatorname{CRPS}(F,y)
=
\int_{-\infty}^{\infty}
\left(
F(t)-\mathbf{1}[y\le t]
\right)^2
dt
$$

where $F$ is the predicted cumulative distribution function.

NLL, Brier score, and CRPS are proper scoring rules. NLL penalizes confident wrong probabilities strongly; Brier score is bounded and often easier to interpret for binary risk.

## Non-Examples

Accuracy, F1, AUROC, and AUPRC are not proper scoring rules. They can be useful metrics, but they do not reward the full predictive distribution.

For example, threshold accuracy evaluates:

$$
\mathbf{1}
\left[
\mathbf{1}[p\ge \tau]=y
\right]
$$

This can be optimized by changing a score around a threshold, even if the reported probability no longer matches the true event probability.

## Proper vs Calibrated

A good proper scoring rule value does not guarantee perfect calibration in every slice. It summarizes probability quality. [[concepts/evaluation/reliability-diagram|Reliability diagrams]] and calibration error show where the predicted probabilities and empirical frequencies disagree.

The practical workflow is:

$$
\text{probability metric}
\rightarrow
\text{calibration diagnostic}
\rightarrow
\text{decision rule}
$$

## Choosing a Scoring Rule

| Prediction type | Proper score | Notes |
|---|---|---|
| Binary probability | Brier, NLL | Brier is bounded; NLL is sharper on overconfidence |
| Multi-class distribution | NLL, multiclass Brier | check class imbalance separately |
| Sequence distribution | token NLL, sequence NLL | report averaging unit and masks |
| Continuous density | NLL, CRPS | check uncertainty calibration |
| Selective prediction | proper score plus coverage/risk | confidence must support abstention |

## Checks

- Is the model output a probability distribution, not only a score?
- Is the metric computed on held-out data under a fixed protocol?
- Is the score averaged over the correct unit: example, token, atom, residue, graph, or sequence?
- Are invalid, padded, or missing targets masked before scoring?
- Are thresholded metrics reported separately from probability metrics?
- Is the claim about decisions, ranking, calibration, or the full distribution?
- Does the chosen scoring rule match the target type?

## Related

- [[concepts/evaluation/probability-metrics|Probability metrics]]
- [[concepts/evaluation/brier-score|Brier score]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/reliability-diagram|Reliability diagram]]
- [[concepts/evaluation/selective-prediction|Selective prediction]]
- [[concepts/machine-learning/probabilistic-prediction|Probabilistic prediction]]
