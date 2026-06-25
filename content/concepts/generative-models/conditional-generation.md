---
title: Conditional Generation
tags:
  - generative-models
  - conditioning
  - controllability
---

# Conditional Generation

Conditional generation models a distribution over outputs $x$ given a condition $c$.

The goal is:

$$
x \sim p_\theta(x \mid c)
$$

$c$ can be text, class label, image, sequence, graph, structure, property target, scaffold, pocket, or another modality.

## Conditioning Mechanisms

- Concatenate condition tokens or features to the input.
- Cross-attend from generated tokens to condition representations.
- Condition a score, noise, velocity, or decoder network.
- Use classifier or classifier-free guidance during sampling.
- Constrain decoding or generation with validity rules.

## Bio-AI Examples

- Generate a molecule conditioned on target property.
- Generate a ligand conditioned on a [[entities/pocket|pocket]].
- Generate protein sequence conditioned on fold or function.
- Complete a structure conditioned on partial coordinates.

## Checks

- What is the condition $c$ and where does it enter the model?
- Is the condition available at deployment time?
- Does stronger conditioning reduce diversity?
- Is controllability evaluated separately from sample quality?
- Could condition labels leak target information across splits?

## Related

- [[concepts/generative-models/guidance|Guidance]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
