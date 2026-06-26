---
title: Model State Contract
tags:
  - machine-learning
  - reproducibility
  - systems
---

# Model State Contract

A model state contract records what must be saved, restored, and compared when a paper, project, or experiment claims a trained model result. A checkpoint is not only weights.

The reproducible training state is:

$$
s_t
=
(\theta_t,\ o_t,\ h,\ r_t,\ d_t,\ c)
$$

where $\theta_t$ are model parameters, $o_t$ is optimizer state, $h$ are hyperparameters, $r_t$ is random state, $d_t$ is data-loader or sampler state, and $c$ is code/config context.

## State Types

| State | Meaning | Example |
| --- | --- | --- |
| Parameters | learned tensors updated by gradient descent | weights, biases, embeddings |
| Buffers | non-gradient model tensors | BatchNorm running mean, running variance |
| Optimizer state | update memory | momentum, Adam first/second moments |
| Scheduler state | learning-rate timeline | warmup step, decay step |
| Hyperparameters | chosen settings, not learned by the update | learning rate, batch size, loss weights |
| Random state | stochastic process position | seed, RNG state, sampler state |
| Data state | order and identity of examples | split version, epoch, shard, cursor |
| Code/config state | execution context | commit, environment, model config |

## Parameter vs Hyperparameter

Parameters are optimized by the training objective:

$$
\theta_{t+1}
=
\theta_t-\eta_t u(g_t,o_t)
$$

Hyperparameters choose how training happens:

$$
h
=
(\eta_0,\lambda_{\mathrm{wd}},B,\text{schedule},\text{loss weights},\ldots)
$$

Changing $h$ creates a different candidate model. If $h$ is selected using validation data, the final test claim must account for that selection boundary.

## Checkpoint Contract

| Field | Required Question |
| --- | --- |
| Model weights | which architecture and parameter tensors are saved? |
| Buffers | are non-gradient state tensors included? |
| Optimizer | can training resume with the same update dynamics? |
| Scheduler | is the current learning rate and schedule position restored? |
| Step count | is the checkpoint indexed by optimizer step, sample count, token count, or epoch? |
| RNG state | can stochastic training or sampling be resumed? |
| Config | are architecture, tokenizer, featurizer, loss, and data paths versioned? |
| Data split | is the train/validation/test split identity recoverable? |
| Artifact link | are metrics, predictions, and logs tied to the checkpoint? |

## Paper Reading Use

When a paper reports a checkpoint or released model, separate:

| Claim | Needed State |
| --- | --- |
| best validation score | selection rule, validation metric, candidate set |
| final test score | selected checkpoint and untouched test protocol |
| fine-tuned result | base model identity, adapter/full-weight state, fine-tuning data |
| reproducibility claim | code version, config, seeds, data split, environment |
| efficiency claim | batch size, precision, hardware, sequence length, step count |

## Failure Modes

- Only weights are saved, but optimizer/scheduler state is needed for resume.
- The best checkpoint is chosen on test performance.
- Metrics are logged without linking to the exact checkpoint.
- Seeds are recorded, but data split or sampler order is not.
- Fine-tuning notes omit whether the base model, adapters, or full weights changed.
- A released model lacks the preprocessing, tokenizer, featurizer, or coordinate policy.

## Related

- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/training-step-accounting|Training step accounting]]
- [[concepts/machine-learning/model-selection|Model selection]]
- [[concepts/machine-learning/hyperparameter-tuning|Hyperparameter tuning]]
- [[concepts/evaluation/seed-variance|Seed variance]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[concepts/systems/experiment-lifecycle|Experiment lifecycle]]
