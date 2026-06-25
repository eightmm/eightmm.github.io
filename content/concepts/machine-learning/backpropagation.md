---
title: Backpropagation
tags:
  - machine-learning
  - optimization
---

# Backpropagation

Backpropagation computes gradients through a composed computation graph using the chain rule. It tells each parameter how a change would affect the final loss.

For a composition:

$$
y = f(g(x))
$$

the chain rule is:

$$
\frac{\partial y}{\partial x}
=
\frac{\partial f}{\partial g}
\frac{\partial g}{\partial x}
$$

For model training:

$$
\nabla_\theta \mathcal{L}
=
\frac{\partial \mathcal{L}}{\partial f_\theta(x)}
\frac{\partial f_\theta(x)}{\partial \theta}
$$

## Checks

- Are invalid or padded elements masked before loss reduction?
- Are gradients zeroed before the next step unless accumulation is intentional?
- Are frozen parameters excluded from gradient updates?
- Are gradient norms monitored when training is unstable?

## Related

- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[infra/gpu-memory|GPU memory]]
