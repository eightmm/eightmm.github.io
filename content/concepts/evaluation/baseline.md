---
title: Baseline
tags:
  - evaluation
  - methodology
---

# Baseline

A baseline is a simple, credible reference method used to judge whether a new method adds value. Without a baseline, improvement claims have no anchor.

An improvement over baseline can be written as:

$$
\Delta M
= M(f_{\mathrm{new}})
- M(f_{\mathrm{base}})
$$

where $M$ is an evaluation metric. The sign of $\Delta M$ depends on whether higher or lower is better.

For lower-is-better metrics such as error, define the direction explicitly:

$$
\Delta_{\mathrm{error}}
=
M(f_{\mathrm{base}})
-
M(f_{\mathrm{new}})
$$

so positive values still mean improvement.

## Good Baselines

- Simple enough to understand.
- Strong enough that beating it is meaningful.
- Evaluated under the same split, metric, and preprocessing.
- Matched to the actual task, not chosen to be weak.

## Baseline Types

- Random or majority baseline: checks whether the task is learnable at all.
- Heuristic baseline: simple rule based on domain knowledge.
- Classical ML baseline: linear model, tree model, nearest neighbor, fingerprint model, or BM25.
- Previous published method: compares to the field's reference point.
- Ablated baseline: removes a component from the proposed system.
- Cheap production baseline: the system that would be used if the new method did not exist.

The best baseline depends on the claim. A paper claiming architecture novelty needs a strong same-data baseline. A project claiming practical utility needs a cheap operational baseline.

## Fairness

Baseline comparisons should hold constant:

$$
(\mathcal{D},\ s,\ M,\ B,\ P)
$$

where $\mathcal{D}$ is data, $s$ is split, $M$ is metric, $B$ is tuning budget, and $P$ is preprocessing. If these differ, the result may compare protocols rather than methods.

## Effect Size

The improvement should be large enough to matter:

$$
|\Delta M|
>
\text{noise from seeds, splits, labels, and measurement}
$$

A statistically detectable but tiny improvement may still be operationally irrelevant, while a large improvement with a weak baseline may still fail to prove novelty.

## Checks

- Is the baseline tuned fairly?
- Does it use the same train/validation/test split?
- Is a cheap domain baseline included, such as fingerprint + tree model or sequence nearest neighbor?
- Does the new method beat the baseline on the metric that matters?
- Is the baseline strong enough for the claim being made?
- Are preprocessing, data filtering, and hyperparameter budget comparable?
- Is the same failure counting rule applied to baseline and new method?
- Is the improvement larger than uncertainty or seed variance?

## Related

- [[concepts/research-methodology/hypothesis|Hypothesis]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/effect-size|Effect size]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/machine-learning/linear-model|Linear model]]
