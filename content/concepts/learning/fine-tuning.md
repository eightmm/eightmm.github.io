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

## Why It Matters

- The standard way to specialize a general pretrained model.
- Parameter-efficient variants cut compute and storage per task.
- Sensitive to learning rate, data size, and catastrophic forgetting.

## Checks

- Full fine-tuning vs. adapter/low-rank methods for the data budget?
- Is the model overfitting a small target set?
- Did adaptation degrade general capabilities learned in pretraining?

## Related

- [[concepts/learning/pretraining|Pretraining]]
- [[concepts/learning/transfer-learning|Transfer learning]]
- [[concepts/learning/instruction-tuning|Instruction tuning]]
- [[concepts/learning/domain-adaptation|Domain adaptation]]
- [[concepts/learning/preference-optimization|Preference optimization]]
- [[concepts/learning/supervised-learning|Supervised learning]]
- [[concepts/learning/imitation-learning|Imitation learning]]
- [[concepts/learning/reward-modeling|Reward modeling]]
