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

## Why It Matters

- Targets fast sampling without the full iterative diffusion loop.
- Can be trained from scratch or distilled from a diffusion model.
- Few-step speed must be weighed against sample fidelity.

## Checks

- Is the self-consistency constraint well enforced across the trajectory?
- Distillation from a teacher vs. standalone training?
- How does one-step quality compare to multi-step sampling?

## Related

- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
