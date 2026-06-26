---
title: Optimization
tags:
  - machine-learning
---

# Optimization

Optimization is the process of updating model parameters to improve an objective. In machine learning, the optimized objective is usually a proxy for the real task.

Gradient descent updates parameters in the direction that locally reduces the objective:

$$
\theta_{t+1}
= \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

Here $\eta$ is the learning rate.

The objective can be viewed as a [[concepts/machine-learning/loss-landscape|Loss landscape]]:

$$
\theta \mapsto \mathcal{L}(\theta)
$$

Optimization follows local information from gradients, while evaluation decides whether that local progress supports the intended generalization claim.

## Core Ideas

- Loss defines the training signal.
- Gradients indicate local directions for parameter updates.
- Learning rate controls update size.
- Batches estimate the objective from subsets of data.
- Optimizer state can change the update beyond raw gradients.
- Curvature and scale affect how stable a step size is.
- Second-order methods or diagnostics use curvature information directly or approximately.
- Constraints may be enforced exactly, approximated by penalties, or handled by projection/decoding.

## Update View

A generic first-order optimizer can be written as:

$$
g_t
=
\nabla_\theta \mathcal{L}(\theta_t)
$$

$$
s_t
=
\operatorname{StateUpdate}(s_{t-1}, g_t)
$$

$$
\theta_{t+1}
=
\operatorname{Update}(\theta_t, g_t, s_t, \eta_t)
$$

For plain gradient descent, $s_t$ is empty. For [[concepts/machine-learning/adam|Adam]] and [[concepts/machine-learning/adamw|AdamW]], $s_t$ contains moment estimates.

## Optimization Protocol

| Component | What To Record |
| --- | --- |
| optimized objective | full loss, auxiliary losses, regularization, constraints, and weights |
| update unit | micro-batch, accumulated batch, optimizer step, epoch, sample, token, graph, or trajectory |
| gradient estimate | batch sampling rule, masking rule, distributed averaging, and valid element denominator |
| optimizer state | momentum, adaptive moments, weight decay, gradient scaler, and scheduler state |
| learning-rate schedule | warmup, decay, restart, step boundary, and resume behavior |
| stability controls | clipping, normalization, mixed precision, loss scaling, skip-step logic |
| selection rule | checkpoint, early stopping metric, validation split, seed policy, and search budget |

The practical update is rarely just raw gradient descent. A useful abstraction is:

$$
\theta_{t+1}
=
\theta_t
-
\eta_t
P_t
\tilde{g}_t
$$

where $\tilde{g}_t$ is the processed gradient after masking, averaging, clipping, scaling, or preconditioning, and $P_t$ represents optimizer-specific scaling such as adaptive moments or approximate curvature.

## Constraints and Penalties

Some optimization problems include a feasible set:

$$
\theta^\*
=
\arg\min_{\theta\in\mathcal{C}}
\mathcal{L}(\theta)
$$

Others use a soft penalty:

$$
\theta^\*
=
\arg\min_\theta
\mathcal{L}(\theta)+\lambda R(\theta)
$$

These are not the same claim. A constraint says the solution must remain valid; a penalty only changes the objective. See [[concepts/math/constrained-optimization|Constrained optimization]].

This distinction matters for constrained decoding, KL-regularized policies, molecular geometry, valid graphs, and resource-constrained model selection.

## Curvature View

First-order methods use gradients. Second-order methods also use curvature:

$$
H_t
=
\nabla^2_\theta \mathcal{L}(\theta_t)
$$

The Hessian is usually too large to materialize for deep models, but Hessian-vector products, diagonal approximations, and curvature diagnostics still help explain unstable training and learning-rate sensitivity.

## Training Signal vs Claim

Optimization only proves that the training procedure reduced the chosen objective under a protocol. It does not automatically prove generalization, calibration, ranking quality, sampling quality, or biological utility.

| Observation | What It Supports | What It Does Not Prove |
| --- | --- | --- |
| train loss decreases | optimizer follows a useful training signal | validation/test improvement |
| validation loss improves | selected objective helps held-out validation distribution | deployment or OOD claim |
| metric improves but loss worsens | surrogate loss and metric may be misaligned | that the loss is wrong without protocol audit |
| unstable loss | step size, gradient scale, precision, data order, or objective issue | architecture failure by itself |
| best checkpoint late/early | selection rule matters | final test can be repeatedly inspected |

## Paper Reading Checklist

| Claim | Required Evidence |
| --- | --- |
| better optimizer | same objective, schedule budget, batch size, step count, and tuning budget |
| faster convergence | wall-clock, optimizer steps, samples/tokens, hardware, and precision all reported |
| better stability | gradient norm, loss scale, NaN/overflow handling, seed variation, and failure rate |
| better final quality | untouched final test after validation-only model selection |
| better scaling | compute, memory, communication, and data pipeline bottlenecks separated |

## Watch For

- Lower training loss does not guarantee better generalization.
- Optimization instability can look like model failure.
- Hyperparameters can dominate small experiments.
- Broken gradients can look like a bad optimizer.
- A scheduler stepped at the wrong boundary can silently change the experiment.
- Gradient accumulation changes the effective batch only if loss scaling and synchronization boundaries are correct.
- Mixed precision can silently skip optimizer steps or change numerical behavior.
- Comparing optimizers without equal tuning budget is usually not a clean optimizer claim.
- A hard constraint, a soft regularizer, and a post-hoc filter support different claims.

## Related

- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/machine-learning/automatic-differentiation|Automatic differentiation]]
- [[concepts/machine-learning/gradient-checking|Gradient checking]]
- [[concepts/machine-learning/loss-landscape|Loss landscape]]
- [[concepts/machine-learning/second-order-optimization|Second-order optimization]]
- [[concepts/math/constrained-optimization|Constrained optimization]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/gradient-accumulation|Gradient accumulation]]
- [[concepts/machine-learning/training-stability|Training stability]]
- [[concepts/machine-learning/training-step-accounting|Training step accounting]]
- [[concepts/machine-learning/model-selection|Model selection]]
- [[concepts/machine-learning/objective-metric-alignment|Objective-metric alignment]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/evaluation/index|Evaluation]]
