---
title: Automatic Differentiation
tags:
  - machine-learning
  - optimization
---

# Automatic Differentiation

Automatic differentiation computes exact derivatives of a program by applying the chain rule to elementary operations. It is different from symbolic differentiation, which manipulates expressions, and finite differences, which approximate derivatives by perturbing inputs.

For a composed function:

$$
y
=
f_k \circ f_{k-1} \circ \cdots \circ f_1(x)
$$

the derivative is a product of local derivatives:

$$
\frac{\partial y}{\partial x}
=
\frac{\partial f_k}{\partial h_{k-1}}
\frac{\partial f_{k-1}}{\partial h_{k-2}}
\cdots
\frac{\partial f_1}{\partial x}
$$

where $h_i=f_i(h_{i-1})$.

## Forward and Reverse Mode

Forward-mode automatic differentiation propagates tangent information from inputs to outputs. It is efficient when the number of inputs is small and the number of outputs is large.

Reverse-mode automatic differentiation propagates adjoints from outputs back to inputs. It is efficient for scalar losses with many parameters, which is the common neural network case.

For a scalar loss $L$ and intermediate variable $h_i$, reverse mode stores:

$$
\bar{h}_i
=
\frac{\partial L}{\partial h_i}
$$

and propagates:

$$
\bar{h}_{i-1}
=
\left(
\frac{\partial h_i}{\partial h_{i-1}}
\right)^\top
\bar{h}_i
$$

In practice, reverse-mode systems compute vector-Jacobian products rather than materializing full Jacobians:

$$
v^\top J
=
v^\top
\frac{\partial f(x)}{\partial x}
$$

This is why [[concepts/machine-learning/backpropagation|Backpropagation]] scales to models with many parameters.

## Computation Graph

A training step builds a graph of operations:

$$
x
\rightarrow
f_\theta(x)
\rightarrow
\mathcal{L}(f_\theta(x), y)
$$

Each operation records enough information to compute its local backward rule. During backward pass, gradients are accumulated at parameters:

$$
g_\theta
\leftarrow
g_\theta
+
\frac{\partial \mathcal{L}}{\partial \theta}
$$

This accumulation matters for shared parameters, recurrent computation, tied embeddings, and [[concepts/machine-learning/gradient-accumulation|gradient accumulation]].

## Practical Notes

- Stop-gradient or detach breaks a path in the computation graph.
- In-place operations can invalidate saved tensors needed for backward.
- Mixed precision may require gradient scaling before backward stability is reliable.
- Non-differentiable operations require surrogate losses, straight-through estimators, relaxation, or policy gradients.
- Memory use often comes from saved activations rather than parameter gradients.

## Checks

- Is the loss a scalar or reduced in the intended way before backward?
- Are masked, padded, or invalid examples excluded before gradients are computed?
- Are detached tensors intentional?
- Are parameters shared, frozen, or excluded from optimizer groups correctly?
- Does [[concepts/machine-learning/gradient-checking|Gradient checking]] pass for custom operations?

## Related

- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/machine-learning/gradient-checking|Gradient checking]]
- [[concepts/math/jacobian-hessian|Jacobian and Hessian]]
- [[concepts/math/matrix-calculus|Matrix calculus]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
