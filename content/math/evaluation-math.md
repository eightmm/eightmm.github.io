---
title: Evaluation Math
tags:
  - math
  - evaluation
---

# Evaluation Math

Evaluation math separates model performance claims from noise, leakage, calibration problems, and benchmark artifacts.

$$
R(f)
=
\mathbb{E}_{(x,y)\sim p_{\mathrm{test}}}
[\mathcal{L}(f(x), y)]
$$

The empirical test score is only an estimate of this target.

## Route Map

| Route | Use For | Start |
| --- | --- | --- |
| Metric definition | what quantity is averaged, ranked, thresholded, or calibrated | [Metric](/concepts/evaluation/metric), [Metric selection](/concepts/evaluation/metric-selection) |
| Benchmark boundary | what population and claim the benchmark represents | [Benchmark intake](/concepts/data/benchmark-intake), [Claim-evidence boundary](/concepts/evaluation/claim-evidence-boundary) |
| Uncertainty | confidence intervals, bootstrap estimates, seed or sampler variance | [Confidence interval](/concepts/evaluation/confidence-interval), [Bootstrap evaluation](/concepts/evaluation/bootstrap-evaluation) |
| Comparison | effect size, paired tests, multiple-comparison risk | [Effect size](/concepts/evaluation/effect-size), [Paired comparison](/concepts/evaluation/paired-comparison), [Multiple comparisons](/concepts/evaluation/multiple-comparisons) |
| Probability quality | calibrated probabilities, reliability, proper scoring rules | [Calibration](/concepts/evaluation/calibration), [Brier score](/concepts/evaluation/brier-score), [Probability metrics](/concepts/evaluation/probability-metrics) |
| Statistical claim | whether a reported difference is larger than noise or selection bias | [Statistical significance](/concepts/evaluation/statistical-significance) |

## Estimate and Uncertainty

For a test set of size $m$, the empirical risk is:

$$
\hat{R}
=
\frac{1}{m}\sum_{j=1}^{m}
\ell_j
$$

where $\ell_j=\mathcal{L}(f(x_j),y_j)$ is the per-example loss.

An approximate standard error is:

$$
\operatorname{SE}(\hat{R})
\approx
\frac{s_\ell}{\sqrt{m}}
$$

where $s_\ell$ is the sample standard deviation of per-example losses. This approximation is weak when examples are dependent, heavily stratified, or selected after model tuning.

## Comparison Map

| Comparison | Preferred Evidence | Risk |
| --- | --- | --- |
| Same examples, two models | paired difference and confidence interval | aggregate scores hide per-example dependence |
| Many seeds | mean, variance, and selection rule | best seed is mistaken for expected performance |
| Many prompts/checkpoints | held-out selection protocol | repeated trials inflate false discoveries |
| Many datasets | per-dataset effect size | average score hides failures on important strata |
| Imbalanced classification | PR-AUC, enrichment, calibrated threshold | ROC-AUC can look good under severe imbalance |

For paired comparison:

$$
\Delta
=
\frac{1}{m}\sum_{j=1}^{m}
(s_{A,j}-s_{B,j})
$$

where $s_{A,j}$ and $s_{B,j}$ are scores or losses for two systems on the same example.

## Probability Quality

Probability-valued outputs need separate checks from accuracy. A classifier can be accurate but badly calibrated, and a low expected loss can still hide unreliable confidence on a shifted domain.

| Question | Start |
| --- | --- |
| Do predicted probabilities match empirical frequencies? | [Calibration](/concepts/evaluation/calibration), [Reliability diagram](/concepts/evaluation/reliability-diagram) |
| Is the probability score proper? | [Brier score](/concepts/evaluation/brier-score), [Probability metrics](/concepts/evaluation/probability-metrics) |
| Is uncertainty part of the decision? | [Uncertainty estimation](/concepts/evaluation/uncertainty-estimation), [Conformal prediction](/concepts/evaluation/conformal-prediction) |

## Reporting Checks

Use the evaluation formula only after the quantities are pinned down.

| Question | Mathematical Object |
| --- | --- |
| What is averaged? | per-example loss, score, binary success, rank statistic, or generated-sample property |
| Over which set? | test examples, queries, targets, scaffolds, proteins, prompts, seeds, or generated samples |
| What is random? | data sampling, model initialization, decoding, environment, assay noise, or bootstrap resampling |
| What is compared? | paired examples, independent aggregates, confidence intervals, or effect sizes |
| What was selected? | checkpoint, prompt, threshold, hyperparameter, preprocessing rule, or model family |

For a reported score $S$, a useful paper note should identify:

$$
S
=
\operatorname{aggregate}
\left(
\{s_j\}_{j=1}^{m};
\text{metric},\ \text{selection rule},\ \text{split}
\right)
$$

where $s_j$ is the per-unit score and the aggregation rule is part of the claim.

## Checks

- Is the reported score a point estimate or an uncertainty-aware comparison?
- Is the test set independent from model selection?
- Are comparisons paired on the same examples?
- Is the metric aligned with the real decision?
- Are confidence, calibration, and abstention needed?

## Related

- [[math/index|Math]]
- [[ai/evaluation|Evaluation]]
- [[molecular-modeling/data-evaluation|Computational Biology data and evaluation]]
