---
title: Probability Flow ODE
tags:
  - generative-models
  - diffusion
  - score-based-model
---

# Probability Flow ODE

A probability flow ODE is a deterministic trajectory whose marginal distributions match those of a corresponding diffusion SDE. It is useful because a score-based model can be sampled by integrating an ODE rather than simulating stochastic reverse diffusion.

For a forward SDE:

$$
d x = f(x,t)\,dt + g(t)\,d w
$$

the probability flow ODE is:

$$
\frac{dx}{dt}
=
f(x,t)
-
\frac{1}{2}g(t)^2\nabla_x \log p_t(x)
$$

$f$ is the drift, $g$ is the diffusion coefficient, $w$ is Brownian motion, and $\nabla_x\log p_t(x)$ is the score at time $t$.

## Why It Matters

- Turns stochastic diffusion sampling into deterministic ODE integration.
- Connects score-based diffusion to continuous normalizing flow views.
- Enables likelihood computation through instantaneous change of variables when the divergence is tractable or estimated.
- Makes solver choice part of the sampling contract.

## Sampling Contract

Sampling starts from noise $x_T \sim p_T$ and integrates backward to $x_0$:

$$
x_0
=
\operatorname{ODESolve}
\left(
x_T,
v_\theta(x,t),
T \rightarrow 0
\right)
$$

where $v_\theta$ is the learned or score-derived velocity field.

The ODE can also be read as deterministic transport:

$$
\frac{dx}{dt}
=
v_\theta(x,t)
$$

where $v_\theta$ may be parameterized directly, as in [[concepts/generative-models/flow-matching|Flow matching]], or derived from a score field:

$$
v_\theta(x,t)
=
f(x,t)
-
\frac{1}{2}g(t)^2 s_\theta(x,t)
$$

This makes the learned object explicit: some papers train a score, others train a velocity, and others train a denoiser that is converted to one of these.

## Checks

- Which ODE solver, tolerance, and number of function evaluations are used?
- Is the score accurate enough at the integration endpoints?
- Does deterministic sampling reduce diversity for the task?
- Are coordinates, graphs, or sequences integrated in a representation that preserves validity?
- Is the comparison fair against stochastic samplers with different compute budgets?
- Is the model trained as score prediction, noise prediction, denoising, or velocity prediction?
- Is likelihood estimation claimed, or only deterministic sampling?

## Related

- [[concepts/generative-models/score-matching|Score matching]]
- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- [[concepts/generative-models/energy-based-model|Energy-based model]]
- [[concepts/generative-models/sampling|Sampling]]
