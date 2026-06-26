---
title: Supervised Learning
tags:
  - supervised-learning
  - machine-learning
---

# Supervised Learning

Supervised learning fits a model to map inputs to known target labels by minimizing a loss over labeled examples. It covers classification, regression, and structured prediction.

The empirical supervised objective is:

$$
\hat{\theta}
= \arg\min_\theta
\frac{1}{n}\sum_{i=1}^{n}
\mathcal{L}(f_\theta(x_i), y_i)
$$

The main assumption is that labels $y_i$ are meaningful for the target task and split.

## Population View

The empirical objective estimates a population risk:

$$
R(f)
=
\mathbb{E}_{(x,y)\sim p_{\mathrm{data}}}
\left[
\mathcal{L}(f(x),y)
\right]
$$

Training minimizes:

$$
\hat{R}_{\mathrm{train}}(f)
=
\frac{1}{n_{\mathrm{train}}}
\sum_{i\in\mathrm{train}}
\mathcal{L}(f(x_i),y_i)
$$

but evaluation cares about held-out or deployment risk:

$$
\hat{R}_{\mathrm{test}}(f)
=
\frac{1}{n_{\mathrm{test}}}
\sum_{i\in\mathrm{test}}
\mathcal{L}(f(x_i),y_i)
$$

The gap between these quantities is where overfitting, leakage, and dataset shift appear.

## Task Types

- Classification predicts a discrete label or class distribution.
- Regression predicts a continuous quantity.
- Ranking predicts an ordering or preference relation.
- Structured prediction predicts sequences, graphs, masks, boxes, poses, or other structured outputs.

The label space determines the loss. A classification problem often uses cross entropy, while regression often uses squared or absolute error:

$$
\mathcal{L}_{\mathrm{CE}}
=
-\sum_{k} y_k\log p_\theta(y=k\mid x)
$$

$$
\mathcal{L}_{\mathrm{MSE}}
=
\lVert f_\theta(x)-y\rVert_2^2
$$

## Label Semantics

Supervised learning is only as clear as its labels. A label should define:

- What the target means.
- Unit and direction, for numeric labels.
- How replicates or conflicting labels are handled.
- Whether labels are point values, censored values, preferences, or noisy observations.
- Whether the label is available at deployment time.

## Why It Matters

- The baseline paradigm when labels are available and trustworthy.
- Directly optimizes the quantity you care about, given a clean loss.
- Provides the downstream signal that pretraining and fine-tuning are evaluated against.
- Provides cheap baselines that can expose whether a complex method is actually useful.

## Failure Modes

- Train/test leakage through duplicated examples, target-derived features, or preprocessing fit on all data.
- Label noise or inconsistent annotation rules.
- Split mismatch: random split used for a deployment claim that needs domain, time, scaffold, or family split.
- Metric mismatch: optimizing a loss that does not track the decision metric.
- Hidden class imbalance or censored labels treated as ordinary labels.

## Checks

- Are labels accurate, consistent, and free of leakage from the inputs?
- Do train and test splits respect the real generalization boundary?
- Is the loss aligned with the metric used at evaluation?
- Was preprocessing fit only on training data?
- Is the example unit and split unit explicit?
- Are uncertainty, ambiguity, or censored labels handled by the loss?

## Related

- [[concepts/machine-learning/classification|Classification]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/machine-learning/ranking|Ranking]]
- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/learning/semi-supervised-learning|Semi-supervised learning]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/transfer-learning|Transfer learning]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/knowledge-distillation|Knowledge distillation]]
- [[concepts/learning/imitation-learning|Imitation learning]]
- [[concepts/learning/active-learning|Active learning]]
