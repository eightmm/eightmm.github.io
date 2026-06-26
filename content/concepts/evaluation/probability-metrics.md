---
title: Probability Metrics
tags:
  - evaluation
  - metrics
  - calibration
---

# Probability Metrics

Probability metrics evaluate [[concepts/machine-learning/probabilistic-prediction|predicted probabilities]], not only hard decisions or rankings. They matter when a model's confidence drives triage, filtering, decision thresholds, uncertainty estimates, abstention, active learning, or downstream planning.

For predictions $p_\theta(y\mid x)$ on test examples $(x_i,y_i)$, negative log-likelihood is:

$$
\operatorname{NLL}
=
-
\frac{1}{n}
\sum_{i=1}^{n}
\log p_\theta(y_i\mid x_i)
$$

This is the same mathematical form as a [[concepts/machine-learning/negative-log-likelihood|negative log-likelihood]] training loss, but in evaluation it is computed on held-out data under a fixed protocol.

For multi-class classification with logits $z_{ik}$:

$$
p_{ik}
=
\operatorname{softmax}(z_i)_k
=
\frac{\exp z_{ik}}{\sum_{j=1}^{K}\exp z_{ij}}
$$

and the per-example log loss is:

$$
\ell_i
=
-
\sum_{k=1}^{K}
y_{ik}\log p_{ik}
$$

where $y_{ik}$ is one-hot or a soft target. In sequence models, the reported NLL must state whether it is averaged per token, residue, atom, graph, sequence, or example.

## What Probability Metrics Measure

| Metric | Measures | Common failure it exposes |
|---|---|---|
| NLL / log loss | probability assigned to the observed target | overconfident wrong predictions |
| Brier score | squared probability error | poor binary risk estimates |
| ECE / reliability | calibration by confidence bins | confidence-frequency mismatch |
| CRPS | full continuous predictive distribution | bad uncertainty in regression |
| Selective risk | risk after abstaining | unusable confidence ranking |

Accuracy, AUROC, and AUPRC can be high even when probability estimates are unusable.

## Brier Score

See [[concepts/evaluation/brier-score|Brier score]] for the standalone note.

For binary labels $y_i\in\{0,1\}$ and predicted probability $p_i=p_\theta(y=1\mid x_i)$:

$$
\operatorname{Brier}
=
\frac{1}{n}
\sum_{i=1}^{n}
(p_i-y_i)^2
$$

For multiclass prediction:

$$
\operatorname{Brier}
=
\frac{1}{n}
\sum_{i=1}^{n}
\sum_{k=1}^{K}
(p_{ik}-y_{ik})^2
$$

where $p_{ik}$ is the predicted probability for class $k$ and $y_{ik}$ is the one-hot or soft target.

The binary Brier score has a useful decomposition:

$$
\operatorname{Brier}
=
\operatorname{Reliability}
-
\operatorname{Resolution}
+
\operatorname{Uncertainty}
$$

Reliability penalizes miscalibration, resolution rewards separating examples into groups with different event rates, and uncertainty is determined by the base rate.

## Proper Scoring Rule

See [[concepts/evaluation/proper-scoring-rule|Proper scoring rule]] for the standalone note.

A proper scoring rule rewards honest probabilities: the expected score is optimized when the predicted distribution equals the true conditional distribution.

For a scoring rule $S(p,y)$, propriety means:

$$
\mathbb{E}_{Y\sim q}
[S(q,Y)]
\le
\mathbb{E}_{Y\sim q}
[S(p,Y)]
$$

for a loss-style score where lower is better. NLL and Brier score are common proper scoring rules.

Improper probability scores can reward distorted confidence. For example, threshold accuracy may favor pushing probabilities away from the true uncertainty if that improves a single operating point.

## Calibration Metrics

Expected calibration error bins predictions by confidence:

$$
\operatorname{ECE}
=
\sum_{b=1}^{B}
\frac{|I_b|}{n}
\left|
\operatorname{acc}(I_b)
-
\operatorname{conf}(I_b)
\right|
$$

where $I_b$ is the set of examples in bin $b$.

For binary events:

$$
\operatorname{acc}(I_b)
=
\frac{1}{|I_b|}
\sum_{i\in I_b} y_i,
\qquad
\operatorname{conf}(I_b)
=
\frac{1}{|I_b|}
\sum_{i\in I_b} p_i
$$

ECE depends on binning. Always pair it with [[concepts/evaluation/reliability-diagram|Reliability diagram]] or bin-level inspection when the claim is calibration.

## Regression Probability Metrics

For continuous targets, a probabilistic model predicts a density:

$$
p_\theta(y\mid x)
$$

The test NLL is still:

$$
\operatorname{NLL}
=
-
\frac{1}{n}
\sum_{i=1}^{n}
\log p_\theta(y_i\mid x_i)
$$

For a Gaussian predictive distribution:

$$
\log p_\theta(y_i\mid x_i)
=
-
\frac{1}{2}
\left[
\log(2\pi\sigma_i^2)
+
\frac{(y_i-\mu_i)^2}{\sigma_i^2}
\right]
$$

This evaluates both location and uncertainty. A model with accurate means but under-estimated variance can have poor NLL.

## Metric, Loss, and Calibration

- Loss: used to update parameters during training.
- Probability metric: computed on validation or test data to evaluate probabilistic predictions.
- [[concepts/evaluation/proper-scoring-rule|Proper scoring rule]]: rewards honest probability estimates.
- [[concepts/evaluation/calibration|Calibration]]: asks whether predicted confidence matches empirical frequency.
- [[concepts/evaluation/threshold-selection|Threshold selection]]: converts probabilities into decisions.
- [[concepts/evaluation/selective-prediction|Selective prediction]]: evaluates accept/abstain behavior as confidence thresholds change.

A model can have good accuracy but poor NLL if it is overconfident on errors. It can also have a strong ranking metric but unreliable probabilities.

## Bio-AI Uses

- Activity screening: probability can represent active/inactive risk only if labels and prevalence are compatible with deployment.
- Structure prediction confidence: per-residue or per-pair probabilities need masking and aggregation rules.
- Generative filtering: probability metrics do not replace validity, novelty, diversity, or downstream property metrics.
- Assay prediction: calibrated uncertainty is useful only if assay heterogeneity and [[concepts/evaluation/assay-harmonization|Assay harmonization]] are handled.

## Checks

- Are probabilities required, or are rankings/classes enough?
- Is NLL averaged per example, token, atom, sequence, or graph?
- Are padding and invalid positions masked consistently?
- Are probabilities calibrated on validation data, not the test set?
- Are threshold-dependent metrics reported separately from probability metrics?
- Is calibration checked globally and within important subgroups?
- Are probability estimates evaluated under the same prevalence and shift expected at use time?
- Is uncertainty inflated, collapsed, or only a monotone confidence score?

## Related

- [[concepts/evaluation/metric|Metric]]
- [[concepts/machine-learning/probabilistic-prediction|Probabilistic prediction]]
- [[concepts/machine-learning/decision-rule|Decision rule]]
- [[concepts/evaluation/proper-scoring-rule|Proper scoring rule]]
- [[concepts/evaluation/brier-score|Brier score]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/regression-metrics|Regression metrics]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/reliability-diagram|Reliability diagram]]
- [[concepts/evaluation/threshold-selection|Threshold selection]]
- [[concepts/evaluation/selective-prediction|Selective prediction]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/machine-learning/negative-log-likelihood|Negative log-likelihood]]
- [[concepts/machine-learning/cross-entropy-loss|Cross-entropy loss]]
