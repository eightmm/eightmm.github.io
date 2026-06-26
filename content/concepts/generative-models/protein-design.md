---
title: Protein Design
tags:
  - protein-design
  - generative-model
  - protein-modeling
---

# Protein Design

Protein design generates sequences and/or structures intended to fold, bind, or function as specified, including de novo backbones and sequence design for a target structure.

One conditional view is:

$$
s \sim p_\theta(s \mid X, c)
$$

where $s$ is a sequence, $X$ is a backbone or structural context, and $c$ is a design condition such as binding, fold, or function.

## Design Targets

| Target | Generated Object | Evidence Boundary |
| --- | --- | --- |
| Sequence design | sequence for a fixed backbone or fold | sequence recovery and predicted structure are not enough for function |
| Backbone design | coordinates or fold scaffold | geometry plausibility does not prove foldability or expression |
| Binder design | sequence, backbone, or complex against a target | docking or predicted complex score is not experimental binding |
| Functional design | sequence or structure with desired activity | proxy model score is not functional validation |
| Motif scaffolding | scaffold around constrained residues or motif | motif placement must be separated from global stability |

## Objective Patterns

Autoregressive sequence design can be written as:

$$
p_\theta(s \mid X, c)
=
\prod_{i=1}^{L}
p_\theta(s_i \mid s_{<i}, X, c)
$$

Diffusion or flow-style structure design often models a path over coordinates:

$$
X_t
\rightarrow
v_\theta(X_t, t, c)
\rightarrow
X_0
$$

where $X_t$ is a noisy or intermediate structure, $v_\theta$ is a denoising or velocity field, and $c$ is the design condition.

## Evaluation Contract

| Check | Why |
| --- | --- |
| Training similarity | near-template or homologous designs can look novel only superficially |
| Structural validity | backbone geometry, clashes, side-chain packing, and fold confidence test different failures |
| Conditional satisfaction | designs must satisfy the requested fold, motif, target, or function |
| Diversity | many high-scoring designs may be near-duplicates |
| Experimental status | in-silico validation must not be written as wet-lab success |
| Negative controls | generated designs should be compared with simple baselines or shuffled conditions |

## Why It Matters

- Enables novel binders, enzymes, and scaffolds beyond natural proteins.
- Combines structure generation with sequence design and validation.
- Computational success must hold up against experimental reality.

## Checks

- Are designs evaluated for foldability and structural consistency?
- Does the model exploit training data instead of generalizing?
- How are designs validated beyond in-silico metrics?
- Is the claim about sequence recovery, structure plausibility, binding, function, or experimental validation?
- Are generated samples filtered by a separate model that owns part of the final claim?

## Related

- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[molecular-modeling/protein-modeling|Protein modeling]]
- [[molecular-modeling/paper-claim-patterns|Computational Biology paper claim patterns]]
- [[concepts/ai-computational-biology-math-contract|AI Computational Biology Math contract]]
