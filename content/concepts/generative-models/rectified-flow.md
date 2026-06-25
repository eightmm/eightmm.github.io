---
title: Rectified Flow
tags:
  - rectified-flow
  - flow-matching
  - generative-model
---

# Rectified Flow

Rectified flow learns a transport that follows straight-line paths between a noise distribution and the data distribution, enabling fast, few-step sampling.

The straight-line interpolation is often:

$$
x_t = (1-t)x_0 + t x_1
$$

with target velocity:

$$
u_t = x_1 - x_0
$$

The model learns a velocity field that approximates this transport.

## Why It Matters

- Straighter trajectories reduce the number of sampling steps.
- Closely related to flow matching and ODE-based generation.
- A practical route to fast diffusion-style samplers.

## Checks

- Are the learned paths actually close to straight?
- How much does reflow/distillation improve few-step quality?
- Does few-step sampling preserve diversity?

## Related

- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/consistency-model|Consistency model]]
