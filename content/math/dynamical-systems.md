---
title: Dynamical Systems
tags:
  - math
  - dynamical-systems
  - generative-models
---

# Dynamical Systems

Dynamical systems describe how a state changes over time or along a continuous path. In AI, this appears in recurrent models, residual networks, diffusion models, flow matching, probability-flow ODEs, control, reinforcement learning, and molecular dynamics.

## Core Notes

- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/probability-flow-ode|Probability flow ODE]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/residual-network|Residual network]]
- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[concepts/molecular-modeling/molecular-dynamics|Molecular dynamics]]

## Discrete-Time Dynamics

A discrete-time system updates state step by step:

$$
x_{t+1}
=
F_\theta(x_t,t)
$$

This view covers recurrent networks, iterative refinement, optimization steps, and some agent state transitions.

## Continuous-Time Dynamics

A continuous-time system is often written as an ODE:

$$
\frac{dx(t)}{dt}
=
f_\theta(x(t),t)
$$

The function $f_\theta$ is a vector field. It tells the direction and speed of movement at each point.

## Flow Map

Solving the ODE from time $s$ to time $t$ defines a flow map:

$$
x(t)
=
\Phi_{s\to t}(x(s))
$$

Generative flows use this idea to move samples from a simple distribution to a data distribution.

## Stochastic Dynamics

Diffusion-style models often include noise. A simplified stochastic differential equation has the form:

$$
dx
=
f(x,t)\,dt
+
g(t)\,dW_t
$$

where $W_t$ is Brownian motion. The drift $f$ controls deterministic movement, and the diffusion term $g(t)$ controls noise scale.

## Residual Networks as Dynamics

A residual block:

$$
h_{l+1}
=
h_l + F_\theta(h_l)
$$

can be read as a discrete approximation to continuous dynamics:

$$
\frac{dh(t)}{dt}
\approx
F_\theta(h(t))
$$

This perspective helps connect deep architectures, stability, and continuous generative models.

## Molecular Dynamics View

Molecular dynamics is a physical dynamical system over coordinates:

$$
m_i\frac{d^2x_i}{dt^2}
=
-\nabla_{x_i}E(X)
$$

The state can include position and velocity. For AI notes, the important distinction is whether dynamics are simulated by a fixed physical model, learned by a neural model, or used only as post hoc analysis.

## Checks

- Is time discrete, continuous, learned, or just an ordering variable?
- Is the state a vector, sequence, graph, coordinate set, or distribution?
- Is the dynamics deterministic or stochastic?
- Does the model learn a score, vector field, transition kernel, or policy?
- Is numerical integration part of the method or only a conceptual analogy?
- For molecular dynamics, are force field, time step, initialization, and analyzed frames stated?

## Related

- [[math/index|Math]]
- [[math/calculus-gradients|Calculus and gradients]]
- [[math/probability-statistics|Probability and statistics]]
- [[math/numerical-computing|Numerical computing]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/molecular-modeling/molecular-dynamics|Molecular dynamics]]
- [[concepts/molecular-modeling/force-field|Force field]]
