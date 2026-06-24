---
title: Rectified Flow
tags:
  - rectified-flow
  - flow-matching
  - generative-model
---

# Rectified Flow

Rectified flow learns a transport that follows straight-line paths between a noise distribution and the data distribution, enabling fast, few-step sampling.

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
