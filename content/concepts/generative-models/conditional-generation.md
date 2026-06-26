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

## Conditional Objective

Most conditional generative objectives optimize a conditional distribution or conditional denoising target:

$$
\mathcal{J}(\theta)
=
\mathbb{E}_{(x,c)\sim p_{\mathrm{data}}}
\left[
-\log p_\theta(x\mid c)
\right]
$$

or, for diffusion/flow-style models:

$$
\mathbb{E}
\left[
\ell_\theta(x_t,t,c,x)
\right].
$$

The condition $c$ is part of the data contract. A paper should state whether $c$ is observed at inference, predicted by another model, retrieved, manually specified, or derived from the target output.

## Conditioning Mechanisms

- Concatenate condition tokens or features to the input.
- Cross-attend from generated tokens to condition representations.
- Condition a score, noise, velocity, or decoder network.
- Use classifier or classifier-free guidance during sampling.
- Constrain decoding or generation with validity rules.

## Condition Boundary

| Condition Type | Example | Leakage Risk |
| --- | --- | --- |
| label or class | molecule property bin, protein family | label may be unavailable or noisy at deployment |
| text or prompt | desired function, natural-language instruction | prompt may encode benchmark answer |
| structure or pocket | binding pocket, partial coordinates | pocket may be ligand-defined or template-derived |
| scaffold or motif | fixed substructure, sequence motif | constraint can make novelty claims narrower |
| retrieval context | nearest examples, database hits | retrieval corpus may include test or homologous examples |
| predicted condition | upstream model output | error propagation must be measured |

## Evaluation Axes

Conditional generation should be evaluated along several axes:

$$
\text{quality}
\ne
\text{condition satisfaction}
\ne
\text{diversity}
\ne
\text{novelty}
$$

| Axis | Question |
| --- | --- |
| validity | is the sample syntactically, chemically, physically, or structurally valid? |
| fidelity | does it satisfy $c$? |
| diversity | are samples varied under the same condition? |
| novelty | are samples not memorized or near-duplicates? |
| utility | does the sample help the downstream task? |
| cost | how much sampling, guidance, filtering, or reranking was used? |

## Molecular Modeling Examples

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
- Is $c$ raw, predicted, retrieved, or derived from the target output?
- Are filtered invalid samples counted in the evaluation?

## Related

- [[concepts/generative-models/guidance|Guidance]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
