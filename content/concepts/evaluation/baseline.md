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

## Good Baselines

- Simple enough to understand.
- Strong enough that beating it is meaningful.
- Evaluated under the same split, metric, and preprocessing.
- Matched to the actual task, not chosen to be weak.

## Checks

- Is the baseline tuned fairly?
- Does it use the same train/validation/test split?
- Is a cheap domain baseline included, such as fingerprint + tree model or sequence nearest neighbor?
- Does the new method beat the baseline on the metric that matters?

## Related

- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/machine-learning/linear-model|Linear model]]
