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

## Conditional Energy

Energy functions can also score a candidate output under a condition:

$$
p_\theta(y\mid x)
=
\frac{\exp(-E_\theta(x,y))}
{Z_\theta(x)}
$$

where:

$$
Z_\theta(x)
=
\sum_{y'} \exp(-E_\theta(x,y'))
$$

or an integral for continuous $y$. This form appears in structured prediction, compatibility scoring, docking-like scoring, and reranking.

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

## Positive and Negative Samples

Many practical EBMs learn by contrasting observed examples with negatives:

$$
\Delta E
=
E_\theta(x^+) - E_\theta(x^-)
$$

The model should assign lower energy to positives:

$$
E_\theta(x^+) < E_\theta(x^-)
$$

Negative sampling defines the task. Easy negatives may make the model look good without learning useful structure; false negatives can corrupt the signal.

| Negative source | Risk |
| --- | --- |
| random samples | too easy |
| corrupted examples | corruption may be unrealistic |
| model samples | expensive or unstable |
| in-batch negatives | unmarked positives may appear |
| hard negatives | improves discrimination but can introduce bias |

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

## Energy Calibration

Low energy is not automatically a calibrated probability, physical energy, or valid score.

| Claim | Need |
| --- | --- |
| ranks candidates | pairwise or ranking evaluation |
| defines likelihood | tractable or estimated partition behavior |
| generates samples | sampling procedure and validity checks |
| approximates physical energy | units, reference state, force/energy benchmark |
| supports decisions | calibration or threshold validation |

For molecular modeling, a learned pseudo-energy should not be presented as a physical potential unless trained and validated for that claim.

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
- Are negatives realistic and free of hidden positives?
- Is energy used as ranking score, probability model, generator, or physical quantity?

## Related

- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/score-matching|Score matching]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/machine-learning/ranking|Ranking]]
- [[concepts/molecular-modeling/force-field|Force field]]
- [[concepts/molecular-modeling/energy-minimization|Energy minimization]]
