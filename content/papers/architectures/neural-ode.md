---
title: Neural Ordinary Differential Equations
aliases:
  - papers/neural-ode
  - papers/neural-ordinary-differential-equations
tags:
  - papers
  - architectures
  - continuous-depth
---

# Neural Ordinary Differential Equations

> The paper reframed depth as continuous-time dynamics parameterized by a neural network.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Neural Ordinary Differential Equations |
| Authors | Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud |
| Year | 2018 |
| Venue | NeurIPS 2018 |
| arXiv | [1806.07366](https://arxiv.org/abs/1806.07366) |
| Status | verified |

## Question

Residual networks can be read as discrete updates to a hidden state. The question was whether a neural network could define the derivative of a hidden state and use a numerical ODE solver as the layer stack.

## Main Claim

Instead of defining a finite sequence of layers, Neural ODE defines hidden-state dynamics:

$$
\frac{dh(t)}{dt}
=
f_\theta(h(t), t)
$$

and computes the output by solving:

$$
h(t_1)
=
h(t_0)
+
\int_{t_0}^{t_1}
f_\theta(h(t), t)\,dt
$$

This gives a continuous-depth architecture whose computation is controlled by an ODE solver.

## Method

| Component | Role |
| --- | --- |
| neural dynamics function | parameterizes the derivative |
| black-box ODE solver | computes hidden state at target time |
| adjoint sensitivity method | backpropagates through the solve |
| continuous normalizing flow | applies ODE dynamics to density modeling |
| adaptive computation | changes solver steps based on dynamics and tolerance |

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Continuous-depth models can replace residual stacks | image and toy experiments | solver cost can be high |
| Adjoint method reduces memory use | training through ODE solves | numerical issues can affect gradients |
| ODE dynamics enable continuous normalizing flows | density modeling experiments | later CNF work changed estimators and solvers |

## Limitations

- Solver tolerance, stiffness, and numerical error become architecture concerns.
- Wall-clock speed can be worse than fixed-depth networks.
- The adjoint method can trade memory for numerical instability.
- Continuous-depth elegance does not automatically improve benchmark accuracy.

## Why It Matters

Neural ODE is the reference paper for continuous-depth networks and for connecting residual architectures, differential equations, and normalizing flows.

## Connections

- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/residual-network|Residual network]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- [[concepts/generative-models/probability-flow-ode|Probability flow ODE]]
- [[papers/architectures/deep-residual-learning|Deep Residual Learning]]
- [[papers/architectures/index|Architecture papers]]
