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

## Why It Matters

Accuracy only evaluates the final class decision. A proper scoring rule evaluates the full predictive distribution:

$$
p_\theta(y\mid x)
$$

This matters when probabilities are used for [[concepts/evaluation/threshold-selection|Threshold selection]], [[concepts/evaluation/calibration|Calibration]], [[concepts/evaluation/selective-prediction|Selective prediction]], active learning, or risk-sensitive decisions.

## Common Examples

Negative log-likelihood:

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

Both are proper scoring rules. NLL penalizes confident wrong probabilities strongly; Brier score is bounded and often easier to interpret for binary risk.

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

## Checks

- Is the model output a probability distribution, not only a score?
- Is the metric computed on held-out data under a fixed protocol?
- Is the score averaged over the correct unit: example, token, atom, residue, graph, or sequence?
- Are invalid, padded, or missing targets masked before scoring?
- Are thresholded metrics reported separately from probability metrics?

## Related

- [[concepts/evaluation/probability-metrics|Probability metrics]]
- [[concepts/evaluation/brier-score|Brier score]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/reliability-diagram|Reliability diagram]]
- [[concepts/machine-learning/probabilistic-prediction|Probabilistic prediction]]
