---
title: Linear Probing
tags:
  - representation-learning
  - evaluation
  - transfer-learning
---

# Linear Probing

Linear probing evaluates a frozen representation by training a simple linear head on top of it. It asks whether the learned features are already linearly usable for a target task.

For an encoder $f_\theta$ and frozen embedding:

$$
z_i = f_\theta(x_i),
\qquad
\theta\ \text{fixed}
$$

a classification probe is:

$$
\hat{p}(y_i=k \mid x_i)
=
\mathrm{softmax}(Wz_i + b)_k
$$

and the probe is trained by:

$$
W^\*, b^\*
=
\arg\min_{W,b}
\frac{1}{n}\sum_{i=1}^{n}
\mathrm{CE}\left(\mathrm{softmax}(Wz_i+b), y_i\right)
+ \lambda \lVert W\rVert_2^2
$$

where $W$ and $b$ are the only trainable probe parameters, $\mathrm{CE}$ is [[concepts/machine-learning/cross-entropy-loss|cross-entropy loss]], and $\lambda$ is a regularization strength selected on validation data.

For regression, the probe can be:

$$
\hat{y}_i = Wz_i + b,
\qquad
\min_{W,b}
\frac{1}{n}\sum_i
\lVert Wz_i + b - y_i\rVert_2^2
$$

## What It Tests

Linear probing tests representation separability, not the full adaptation capacity of the pretrained model.

It is useful when the question is:

- Does pretraining organize examples so that the target is easy to read out?
- Does a representation contain task-relevant information before task-specific adaptation?
- Does a new SSL objective improve frozen features under the same protocol?

It is weaker when the downstream task requires nonlinear reasoning, task-specific alignment, or structural decoding.

## Protocol

- Freeze the encoder before fitting the probe.
- Fit preprocessing and normalization only on the training split.
- Select probe regularization, learning rate, batch size, and checkpoint using validation data.
- Report the untouched test result only once after model and probe choices are fixed.
- Compare against simple feature baselines, random encoder baselines, and [[concepts/learning/fine-tuning-protocol|fine-tuning protocol]] when relevant.

## Failure Modes

- Hidden tuning of the encoder on the probe validation set.
- Comparing probes with different search budgets.
- Using a nonlinear head but still calling it a linear probe.
- Splitting after embedding cache construction, causing duplicate or source leakage.
- Claiming downstream transfer from a probe that only tests interpolation.

## Checks

- What layer, token, pooling rule, or graph readout defines $z$?
- Is the probe exactly linear after the frozen representation?
- What hyperparameters were selected on validation data?
- What split unit supports the generalization claim?
- Is the result compared to full fine-tuning and a simple non-neural baseline?

## Related

- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/learning/fine-tuning-protocol|Fine-tuning protocol]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/leakage|Leakage]]
