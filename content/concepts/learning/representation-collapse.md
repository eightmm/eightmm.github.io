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

## Diagnostics

Collapse should be measured, not guessed from the pretraining loss.

| Diagnostic | What it checks | Failure signal |
| --- | --- | --- |
| per-dimension variance | whether each embedding dimension carries variation | many dimensions near zero variance |
| covariance spectrum | whether representation uses multiple directions | one or few dominant eigenvalues |
| pairwise distance distribution | whether examples separate in embedding space | distances concentrate near zero |
| nearest neighbors | whether similarity is semantically meaningful | unrelated examples become nearest neighbors |
| linear probe or kNN | whether representation transfers | no gain over simple baseline |

For embeddings $Z\in\mathbb{R}^{n\times d}$, a simple variance diagnostic is:

$$
\sigma_j^2
=
\frac{1}{n}\sum_{i=1}^{n}
(z_{ij}-\bar{z}_j)^2
$$

and a covariance diagnostic is:

$$
C
=
\frac{1}{n-1}(Z-\bar{Z})^\top(Z-\bar{Z})
$$

If many $\sigma_j^2$ are near zero or the spectrum of $C$ is nearly rank-one, the representation may be collapsed or highly anisotropic.

## Collapse vs Useful Invariance

Not all removed variation is bad. A good representation can ignore nuisance factors while preserving task-relevant factors.

| Case | Interpretation |
| --- | --- |
| low variance across augmented views, high variance across examples | desired invariance |
| low variance across both views and examples | collapse |
| high variance but poor downstream probe | variation exists but may be irrelevant |
| good train probe, poor split transfer | shortcut or leakage-sensitive representation |

## Checks

- What is the batch-wise variance of embeddings?
- Do nearest neighbors contain meaningful semantic or structural similarity?
- Does a [[concepts/learning/linear-probing|linear probe]] perform above a simple baseline?
- Does collapse appear only for a specific augmentation policy, masking ratio, or target encoder update?
- Are variance and covariance measured on held-out data, not only training batches?
- Does the representation preserve variation relevant to the target split unit?

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/learning/augmentation-policy|Augmentation policy]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
