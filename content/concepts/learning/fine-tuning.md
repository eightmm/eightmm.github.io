---
title: Fine-Tuning
tags:
  - fine-tuning
  - transfer-learning
  - machine-learning
---

# Fine-Tuning

Fine-tuning adapts a pretrained model to a target task by continuing training on task data, either updating all weights or a small subset (parameter-efficient fine-tuning).

The basic objective is supervised training initialized from pretrained weights:

$$
\theta^\*
= \arg\min_\theta
\frac{1}{n}\sum_{i=1}^{n}
\mathcal{L}(f_\theta(x_i), y_i),
\qquad
\theta_0 = \theta_{\mathrm{pretrained}}
$$

## Trainable Scope

Fine-tuning is not one protocol. The trainable parameter set can vary:

$$
\Theta_{\mathrm{trainable}}
\subseteq
\Theta_{\mathrm{model}}
$$

Common choices include:

- Head-only tuning.
- Last-layer or last-block tuning.
- Full-model tuning.
- Adapter tuning.
- Low-rank adaptation.
- Prompt or prefix tuning.

The optimization problem is:

$$
\theta^\*
=
\arg\min_{\theta\in\Theta_{\mathrm{trainable}}}
\hat{R}_{\mathrm{train}}(f_\theta)
$$

Reporting fine-tuning without trainable scope is underspecified.

## Learning Rate and Forgetting

Fine-tuning changes the pretrained representation. A large update can improve target performance while degrading broad capabilities:

$$
\Delta\theta
=
\theta_{\mathrm{ft}}-\theta_{\mathrm{pre}}
$$

Catastrophic forgetting risk grows when target data is small, narrow, or distribution-shifted from pretraining:

$$
\operatorname{risk}_{\mathrm{forget}}
\propto
\frac{\lVert\Delta\theta\rVert}{n_{\mathrm{target}}}
$$

This is only a heuristic, but it captures the habit: small target sets usually need conservative learning rates, regularization, early stopping, or parameter-efficient updates.

## Selection Rule

The final checkpoint should be selected by validation data:

$$
t^\*
=
\arg\min_t \hat{R}_{\mathrm{val}}(f_{\theta_t})
$$

Then the test set is used once:

$$
\hat{R}_{\mathrm{test}}(f_{\theta_{t^\*}})
$$

If test performance guides checkpoint selection, the reported result is no longer a clean test estimate.

## Why It Matters

- The standard way to specialize a general pretrained model.
- Parameter-efficient variants cut compute and storage per task.
- Sensitive to learning rate, data size, and catastrophic forgetting.
- Strong results can come from pretraining, adaptation budget, or validation leakage, so the protocol must be explicit.

## Failure Modes

- Comparing methods with different tuning budgets.
- Reporting the best seed without a fixed seed policy.
- Tuning preprocessing or threshold choices on test data.
- Freezing the encoder for one model but fully tuning another without stating parameter counts.
- Forgetting general capability while improving a narrow validation set.
- Overfitting a small target dataset after many hyperparameter trials.

## Checks

- Full fine-tuning vs. adapter/low-rank methods for the data budget?
- Is the model overfitting a small target set?
- Did adaptation degrade general capabilities learned in pretraining?
- Is the adaptation protocol recorded with trainable scope, validation rule, search budget, and final test boundary?
- Are trainable parameter count, optimizer, schedule, seed policy, and early stopping rule reported?
- Is the target split independent from pretraining data, source data, and model-selection choices?
- Is performance compared to linear probing and a simple supervised baseline?

## Related

- [[concepts/learning/pretraining|Pretraining]]
- [[concepts/learning/transfer-learning|Transfer learning]]
- [[concepts/learning/fine-tuning-protocol|Fine-tuning protocol]]
- [[concepts/learning/linear-probing|Linear probing]]
- [[concepts/learning/knowledge-distillation|Knowledge distillation]]
- [[concepts/learning/instruction-tuning|Instruction tuning]]
- [[concepts/learning/domain-adaptation|Domain adaptation]]
- [[concepts/learning/preference-optimization|Preference optimization]]
- [[concepts/learning/supervised-learning|Supervised learning]]
- [[concepts/learning/imitation-learning|Imitation learning]]
- [[concepts/learning/reward-modeling|Reward modeling]]
- [[concepts/machine-learning/early-stopping|Early stopping]]
- [[concepts/machine-learning/hyperparameter-tuning|Hyperparameter tuning]]
