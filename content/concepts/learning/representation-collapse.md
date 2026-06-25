---
title: Representation Collapse
tags:
  - learning
  - self-supervised-learning
  - representation-learning
---

# Representation Collapse

Representation collapse happens when a model maps many different inputs to the same or nearly same embedding. The pretraining loss may look small, but the representation becomes useless for downstream tasks.

A collapsed encoder has low embedding variance:

$$
z_i = f_\theta(x_i),
\qquad
\operatorname{Var}_{i}(z_i) \approx 0
$$

In the extreme case:

$$
f_\theta(x_i)=c
\quad
\text{for all } i
$$

where $c$ is a constant vector.

## Why It Happens

- The objective admits a trivial constant solution.
- Positive-pair alignment is used without a mechanism that preserves information.
- The target network changes too quickly or receives gradients that allow both sides to collapse.
- Augmentations remove too much information.

## Common Prevention Mechanisms

- Negative examples, as in [[concepts/learning/contrastive-learning|contrastive learning]].
- Stop-gradient or EMA target encoders, as in many joint-embedding methods.
- Variance, covariance, or whitening penalties.
- Predictor asymmetry between online and target branches.
- Reconstruction or masked prediction targets that require input-specific information.

## Checks

- What is the batch-wise variance of embeddings?
- Do nearest neighbors contain meaningful semantic or structural similarity?
- Does a [[concepts/learning/linear-probing|linear probe]] perform above a simple baseline?
- Does collapse appear only for a specific augmentation policy, masking ratio, or target encoder update?

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/learning/augmentation-policy|Augmentation policy]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
