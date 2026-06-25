---
title: Dropout
tags:
  - architectures
  - regularization
---

# Dropout

Dropout is a regularization method that randomly removes activations during training. It discourages a model from relying too heavily on any single hidden unit or path.

With keep probability $q=1-p$:

$$
m_i \sim \operatorname{Bernoulli}(q)
$$

$$
\tilde{h}
= \frac{m \odot h}{q}
$$

The scaling keeps the expected activation size similar during training.

## Checks

- Dropout is usually active during training and disabled during inference.
- Too much dropout can underfit.
- Placement matters: attention dropout, residual dropout, feature dropout, and classifier dropout are not identical.
- In small scientific datasets, dropout is only one part of regularization.

## Related

- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
