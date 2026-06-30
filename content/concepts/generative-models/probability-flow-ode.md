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

The deterministic velocity is:

$$
v(x,t)
=
f(x,t)
-
\frac{1}{2}g(t)^2 s_\theta(x,t)
$$

where $s_\theta(x,t)\approx\nabla_x\log p_t(x)$. This makes probability flow ODEs a bridge between score models and flow-like deterministic transport.

## Why It Matters

- Turns stochastic diffusion sampling into deterministic ODE integration.
- Connects score-based diffusion to continuous normalizing flow views.
- Enables likelihood computation through instantaneous change of variables when the divergence is tractable or estimated.
- Makes solver choice part of the sampling contract.

## Change of Variables

For an ODE:

$$
\frac{dx}{dt}=v_\theta(x,t)
$$

the log density evolves as:

$$
\frac{d}{dt}\log p_t(x_t)
=
-\nabla_x\cdot v_\theta(x_t,t)
$$

where $\nabla_x\cdot v$ is the divergence of the velocity field. Integrating this term gives a likelihood route:

$$
\log p_0(x_0)
=
\log p_T(x_T)
+
\int_0^T
\nabla_x\cdot v_\theta(x_t,t)\,dt
$$

depending on time direction convention. The sign should be checked in each paper's notation.

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

## ODE Solver Boundary

The sampler is not only the learned model:

$$
\text{sample}
=
\operatorname{ODESolve}(v_\theta,\ x_T,\ \text{solver},\ \text{tolerance},\ \text{NFE})
$$

where NFE is the number of function evaluations. Quality, speed, and diversity comparisons are weak unless solver settings are fixed or reported.

| Solver Choice | Claim Affected |
| --- | --- |
| NFE / step count | speed-quality tradeoff |
| adaptive tolerance | wall time and numerical error |
| time discretization | endpoint detail and stability |
| stochastic vs deterministic | sample diversity |
| divergence estimator | likelihood accuracy |

## Checks

- Which ODE solver, tolerance, and number of function evaluations are used?
- Is the score accurate enough at the integration endpoints?
- Does deterministic sampling reduce diversity for the task?
- Are coordinates, graphs, or sequences integrated in a representation that preserves validity?
- Is the comparison fair against stochastic samplers with different compute budgets?
- Is the model trained as score prediction, noise prediction, denoising, or velocity prediction?
- Is likelihood estimation claimed, or only deterministic sampling?
- Is the sign convention for forward/backward integration stated?
- Is divergence computed exactly, approximated, or not used?

## Related

- [[concepts/generative-models/score-matching|Score matching]]
- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- [[concepts/generative-models/energy-based-model|Energy-based model]]
- [[concepts/generative-models/sampling|Sampling]]
- [[math/dynamical-systems|Dynamical systems]]
