---
title: Diffusion Model
tags:
  - diffusion-model
  - generative-model
  - machine-learning
---

# Diffusion Model

A diffusion model learns to reverse a gradual noising process: it is trained to denoise corrupted data and generates samples by iteratively denoising from noise.

## Why It Matters

- Strong sample quality and stable training across images, molecules, and proteins.
- Flexible conditioning enables guided and inverse-problem generation.
- Iterative sampling is the main cost, motivating faster samplers.

## Checks

- Does the noise schedule fit the data scale?
- Is conditioning/guidance trading diversity for fidelity?
- How many sampling steps are needed for acceptable quality?

## Related

- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/consistency-model|Consistency model]]
