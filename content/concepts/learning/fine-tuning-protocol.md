---
title: Fine-Tuning Protocol
tags:
  - fine-tuning
  - transfer-learning
  - evaluation
---

# Fine-Tuning Protocol

A fine-tuning protocol specifies how a pretrained model is adapted to a target task and how that adaptation is evaluated. It is broader than saying "fine-tune the model": it includes trainable parameters, optimizer, schedule, validation rule, and final test boundary.

Starting from pretrained weights:

$$
\theta_0 = \theta_{\mathrm{pre}},
\qquad
\theta^\*
=
\arg\min_{\theta\in\Theta_{\mathrm{trainable}}}
\frac{1}{n}\sum_{i=1}^{n}
\mathcal{L}(f_\theta(x_i), y_i)
$$

where $\Theta_{\mathrm{trainable}}$ may include all weights, only a prediction head, adapters, low-rank parameters, or selected layers.

For checkpoint selection:

$$
t^\*
=
\arg\min_t
\hat{R}_{\mathrm{val}}(f_{\theta_t}),
\qquad
\hat{R}_{\mathrm{test}}(f_{\theta_{t^\*}})
\ \text{is reported once}
$$

The validation set chooses the checkpoint. The test set estimates the fixed choice.

## Choices to Record

- Initialization: model version, pretrained objective, and public source.
- Trainable scope: full model, head-only, adapters, low-rank modules, or selected blocks.
- Input contract: preprocessing, tokenization, graph construction, coordinate handling, and missing values.
- Optimizer: learning rate, weight decay, batch size, gradient accumulation, clipping, and schedule.
- Selection rule: validation metric, early stopping, checkpoint cadence, and seed policy.
- Final test: split unit, metric, confidence interval, and failure counting rule.

## Why It Matters

Fine-tuning changes both the representation and the decision head. A gain over [[concepts/learning/linear-probing|linear probing]] may mean that the representation transfers but needs task adaptation, or it may mean the model memorized a small target split.

A fair comparison keeps the adaptation budget visible:

$$
\mathrm{budget}
=
(\text{trainable parameters},\ \text{target labels},\ \text{optimizer steps},\ \text{search trials})
$$

Two methods with different budgets are not evidence-equivalent.

## Failure Modes

- Reusing the test set for checkpoint or hyperparameter selection.
- Reporting the best seed rather than a fixed seed policy.
- Giving one method a larger tuning budget.
- Updating preprocessing parameters with validation or test data.
- Comparing full fine-tuning to a frozen baseline without stating parameter counts.
- Claiming OOD transfer from a split that only tests near-duplicate interpolation.

## Checks

- Which parameters are trainable?
- What validation rule selects the final checkpoint?
- Is the test set untouched until the final fixed run?
- Are compute and search budgets comparable across methods?
- Does the split rule match the transfer claim?

## Related

- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/linear-probing|Linear probing]]
- [[concepts/learning/transfer-learning|Transfer learning]]
- [[concepts/machine-learning/model-selection|Model selection]]
- [[concepts/machine-learning/early-stopping|Early stopping]]
- [[concepts/machine-learning/hyperparameter-tuning|Hyperparameter tuning]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
