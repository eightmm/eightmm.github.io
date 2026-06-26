---
title: Energy-Based Model
tags:
  - generative-models
  - energy-based-model
  - probability
---

# Energy-Based Model

An energy-based model defines a probability distribution through an energy function. Low energy means high probability.

The unnormalized density is:

$$
p_\theta(x)
=
\frac{\exp(-E_\theta(x))}
{Z_\theta}
$$

where the partition function is:

$$
Z_\theta
=
\int \exp(-E_\theta(x))\,dx
$$

For discrete data, the integral becomes a sum over states.

## Energy, Score, and Force

The score of an energy-based distribution is:

$$
\nabla_x \log p_\theta(x)
=
-\nabla_x E_\theta(x)
$$

because $Z_\theta$ does not depend on $x$. This connects energy-based models to [[concepts/generative-models/score-based-model|Score-based model]] and [[concepts/generative-models/score-matching|Score matching]].

In molecular modeling, a force field often uses the sign convention:

$$
F(X)
=
-\nabla_X E(X)
$$

so the learned score, negative energy gradient, and physical force are closely related mathematically, even when their training data and interpretation differ.

## Training Difficulty

Maximum likelihood requires:

$$
\nabla_\theta \log p_\theta(x)
=
-\nabla_\theta E_\theta(x)
-
\nabla_\theta \log Z_\theta
$$

The partition term is:

$$
\nabla_\theta \log Z_\theta
=
\mathbb{E}_{x'\sim p_\theta}
\left[
-\nabla_\theta E_\theta(x')
\right]
$$

This expectation over model samples is often expensive. Practical methods use contrastive divergence, negative sampling, score matching, noise-contrastive estimation, or diffusion-style denoising objectives.

## Sampling

Sampling can be done by Langevin dynamics:

$$
x_{t+1}
=
x_t
-
\eta \nabla_x E_\theta(x_t)
+
\sqrt{2\eta}\,\epsilon_t,
\qquad
\epsilon_t\sim\mathcal{N}(0,I)
$$

This moves samples toward lower energy while retaining stochastic exploration.

## When It Appears

| Context | Energy means | Main caveat |
|---|---|---|
| classical molecular modeling | physical potential energy | force-field assumptions |
| learned molecular models | learned compatibility or pseudo-energy | calibration and OOD validity |
| contrastive representation learning | compatibility between paired objects | negative sampling defines the objective |
| score-based generation | gradient of log density | score may not integrate to a valid global energy |
| structured prediction | cost of an output structure | inference may require optimization |

## Paper Reading Checks

- Is $E_\theta(x)$ a physical energy, learned score, compatibility function, or cost?
- Is the partition function tractable, approximated, or avoided?
- How are negative samples generated?
- Does sampling use Langevin, MCMC, ODE/SDE sampling, or deterministic optimization?
- Are low-energy samples valid under task constraints?
- In molecular settings, is energy evaluated before or after [[concepts/molecular-modeling/energy-minimization|Energy minimization]]?

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/score-matching|Score matching]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/molecular-modeling/force-field|Force field]]
- [[concepts/molecular-modeling/energy-minimization|Energy minimization]]
