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

## Trainable Scope

Different fine-tuning choices test different claims.

| Scope | Trainable Parameters | Claim |
| --- | --- | --- |
| frozen encoder + linear head | small probe only | representation already exposes task signal |
| head + selected layers | partial adaptation | useful lower features with task-specific upper layers |
| full fine-tuning | all weights | model can adapt with target data and budget |
| adapters | small inserted modules | parameter-efficient task adaptation |
| low-rank updates | low-rank trainable matrices | compact adaptation with controlled parameter count |
| prompt or prefix tuning | input-side or prefix parameters | behavior adaptation without full weight update |

The trainable parameter count should be reported:

$$
\rho
=
\frac{\#\Theta_{\mathrm{trainable}}}
{\#\Theta_{\mathrm{total}}}
$$

This makes head-only, adapter, and full fine-tuning comparisons less ambiguous.

## Search Budget

Fine-tuning performance is sensitive to hyperparameter search:

$$
\mathcal{B}
=
(\#\text{seeds},\ \#\text{learning rates},\ \#\text{decay values},\ \#\text{epochs},\ \#\text{protocol variants})
$$

If one method receives a larger $\mathcal{B}$, a better test score may reflect search budget rather than learning method quality.

## Data Boundary

Target data should be split before preprocessing choices that can leak labels or distribution information:

$$
\mathcal{D}
=
\mathcal{D}_{\mathrm{train}}
\cup
\mathcal{D}_{\mathrm{val}}
\cup
\mathcal{D}_{\mathrm{test}}
$$

with:

$$
\mathcal{D}_{\mathrm{train}}
\cap
\mathcal{D}_{\mathrm{test}}
=
\varnothing
$$

The actual split unit may need to be stronger than row identity: scaffold, protein family, target, document, query, patient, or time.

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
- Reporting the best checkpoint after looking at test metrics.
- Changing augmentation, tokenization, graph construction, or conformer generation after validation feedback without recording it.

## Reporting Template

| Field | Record |
| --- | --- |
| initialization | checkpoint identity, pretraining objective, public source |
| trainable scope | full, head-only, adapters, LoRA, selected blocks |
| parameter ratio | $\rho$ trainable / total |
| target data | example unit, split unit, label semantics |
| optimizer | optimizer, learning rate, schedule, weight decay, clipping |
| selection | validation metric, early stopping, seed policy |
| final evaluation | test metric, confidence interval, failure handling |
| public boundary | no private paths, server names, internal task IDs, or unpublished sensitive results |

## Checks

- Which parameters are trainable?
- What validation rule selects the final checkpoint?
- Is the test set untouched until the final fixed run?
- Are compute and search budgets comparable across methods?
- Does the split rule match the transfer claim?
- Are trainable parameter count and search budget reported?
- Are preprocessing and data construction fixed before final test?
- Is the result compared against both a frozen probe and a reasonable from-scratch or simple baseline when relevant?

## Related

- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[papers/architectures/lora|LoRA]]
- [[concepts/learning/linear-probing|Linear probing]]
- [[concepts/learning/transfer-learning|Transfer learning]]
- [[concepts/learning/pretraining|Pretraining]]
- [[concepts/machine-learning/model-selection|Model selection]]
- [[concepts/machine-learning/early-stopping|Early stopping]]
- [[concepts/machine-learning/hyperparameter-tuning|Hyperparameter tuning]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/weight-decay|Weight decay]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
