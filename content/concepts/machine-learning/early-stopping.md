---
title: Early Stopping
tags:
  - machine-learning
  - regularization
  - evaluation
---

# Early Stopping

Early stopping chooses a training checkpoint before the training objective is fully minimized. It is often used as regularization because continued training can reduce training loss while hurting validation performance.

Let $\theta_t$ be model parameters after step or epoch $t$. A validation-based stopping rule chooses:

$$
t^\*
=
\arg\min_{t \in \{1,\ldots,T\}}
\hat{R}_{\mathrm{val}}(f_{\theta_t})
$$

The selected model is:

$$
f^\*
=
f_{\theta_{t^\*}}
$$

This means early stopping is [[concepts/machine-learning/model-selection|model selection]] over checkpoints.

## Patience Rule

A common practical rule stops after validation performance has not improved for $p$ evaluations:

$$
\hat{R}_{\mathrm{val}}(t)
>
\min_{\tau < t}
\hat{R}_{\mathrm{val}}(\tau)
-
\delta
$$

for $p$ consecutive validation checks, where $\delta$ is the minimum improvement threshold. The best checkpoint is usually the previous checkpoint with minimum validation risk.

## What It Regularizes

Early stopping limits the effective training trajectory. In simple settings, it can behave like a complexity constraint: fewer steps often produce smoother or lower-norm solutions than full convergence.

It does not fix:

- Leakage between train and validation.
- A validation split that does not match the target generalization claim.
- Bad labels, wrong loss, or weak features.
- Test-set overuse after checkpoint selection.

## Logging Requirements

Record:

- Validation metric used for stopping.
- Validation frequency and patience.
- Minimum improvement threshold.
- Best checkpoint step or epoch.
- Whether scheduler, optimizer, and random state are saved.
- Whether the final test used the selected checkpoint exactly once.

## Checks

- Is the stopping metric chosen before training?
- Is validation computed on a split that was not used for gradient updates?
- Are preprocessing and augmentation modes correct for validation?
- Is the best checkpoint chosen without test metrics?
- Does the checkpoint include enough [[concepts/systems/checkpoint-state|checkpoint state]] to resume or audit the run?
- Are multiple training attempts counted as part of the selection budget?

## Related

- [[concepts/machine-learning/model-selection|Model selection]]
- [[concepts/machine-learning/hyperparameter-tuning|Hyperparameter tuning]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/machine-learning/overfitting-underfitting|Overfitting and underfitting]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
