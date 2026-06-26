---
title: Consistency Model
tags:
  - consistency-model
  - diffusion-model
  - generative-model
---

# Consistency Model

A consistency model maps any point on a diffusion/probability-flow trajectory directly to its origin, enabling one-step or few-step generation.

The self-consistency condition is:

$$
f_\theta(x_t, t)
= f_\theta(x_s, s)
$$

for two points $x_t$ and $x_s$ on the same trajectory. The model learns a time-consistent map back to the clean sample.

A common training loss compares two trajectory points with a stop-gradient target:

$$
\mathcal{L}_{\mathrm{CM}}
=
\mathbb{E}
\left[
d\left(
f_\theta(x_t,t),
\operatorname{sg}(f_{\theta^-}(x_s,s))
\right)
\right]
$$

where $d$ is a distance such as squared error, $\operatorname{sg}$ stops gradients, and $\theta^-$ may be an exponential-moving-average teacher.

## Sampling View

The model learns a map from noisy states to a clean endpoint:

$$
\hat{x}_0 = f_\theta(x_t,t)
$$

This enables one-step sampling from high noise, or few-step sampling by jumping across a small schedule of noise levels.

## Distillation Contract

Consistency models are often evaluated as acceleration methods. Record:

| Field | Question |
| --- | --- |
| Teacher | Is there a diffusion, score, or flow teacher? |
| Time pairing | How are $t$ and $s$ sampled along the trajectory? |
| Target network | Is the target EMA, frozen teacher, or same model with stop-gradient? |
| Step budget | Is the reported sample one-step, few-step, or adaptive? |
| Guidance | Is classifier-free or other guidance used, and with what scale? |
| Filtering | Are invalid or low-quality samples filtered before reporting? |

The fair comparison is:

$$
\text{quality}(\text{CM}, N_{\mathrm{steps}})
\quad \text{vs.} \quad
\text{quality}(\text{teacher}, N_{\mathrm{steps}})
$$

not only the best teacher quality at many steps against the fastest consistency sample.

## Consistency Error

For two points on the same trajectory:

$$
e_{s,t}
=
\left\|
f_\theta(x_t,t)-f_\theta(x_s,s)
\right\|
$$

Large consistency error at particular noise levels can explain failures in one-step or few-step generation.

## Computational Biology Use

| Use Case | Extra Constraint |
| --- | --- |
| molecular generation | chemical validity and duplicate filtering must be counted |
| conformer generation | bond lengths, chirality, torsions, and energy checks |
| protein design | sequence validity plus structure/function evidence |
| coordinate generation | equivariance and coordinate-frame policy |
| docking pose generation | pose quality, interaction recovery, receptor state |

For structure-based tasks, fast sampling is only useful if geometry validity survives the reduced step count.

## Why It Matters

- Targets fast sampling without the full iterative diffusion loop.
- Can be trained from scratch or distilled from a diffusion model.
- Few-step speed must be weighed against sample fidelity.

## Failure Modes

- One-step samples can lose fine detail or diversity.
- Distillation can inherit teacher biases and evaluation gaps.
- Consistency over sparse time pairs may not cover the full trajectory.
- Speed comparisons are misleading if sample filtering or guidance differs.

## Checks

- Is the self-consistency constraint well enforced across the trajectory?
- Distillation from a teacher vs. standalone training?
- How does one-step quality compare to multi-step sampling?
- Is the sampling step count fixed in evaluation?
- Are diversity and validity measured after acceleration?
- Is speed reported together with preprocessing, filtering, and guidance cost?
- Are failures concentrated at particular noise levels or object types?

## Related

- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/evaluation/paired-comparison|Paired comparison]]
