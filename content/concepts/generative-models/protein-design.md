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

Protein design can also generate coordinates:

$$
X \sim p_\theta(X \mid c)
$$

or joint sequence-structure objects:

$$
(s,X) \sim p_\theta(s,X \mid c)
$$

The claim changes depending on whether the model generates sequence, backbone, side chains, complex pose, or only ranks candidates.

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

Many workflows add filtering or guidance:

$$
\tilde{x}\sim p_\theta(x\mid c),
\qquad
x^\star
=
\arg\max_{\tilde{x}\in \mathcal{C}}
r_\psi(\tilde{x},c)
$$

where $r_\psi$ can be a folding model, structure predictor, docking score, energy model, classifier, or human-defined rule. The reported claim must state whether quality comes from the generator, the filter, or both.

## Claim Ladder

| Claim | Evidence needed |
| --- | --- |
| sequence is plausible | composition, language-model score, diversity, training similarity |
| structure is plausible | backbone geometry, clashes, local quality, fold confidence |
| sequence folds to intended structure | independent structure prediction or experimental structure |
| binder is plausible | interface geometry, negative controls, target specificity checks |
| function is plausible | assay-relevant model or experiment, not only fold confidence |
| experimentally validated | wet-lab evidence with conditions and failure denominator |

## Evaluation Contract

| Check | Why |
| --- | --- |
| Training similarity | near-template or homologous designs can look novel only superficially |
| Structural validity | backbone geometry, clashes, side-chain packing, and fold confidence test different failures |
| Conditional satisfaction | designs must satisfy the requested fold, motif, target, or function |
| Diversity | many high-scoring designs may be near-duplicates |
| Experimental status | in-silico validation must not be written as wet-lab success |
| Negative controls | generated designs should be compared with simple baselines or shuffled conditions |
| Candidate accounting | attempted, filtered, failed, and retained designs must be counted separately |
| Split boundary | training similarity, homologs, templates, and target-family leakage must be checked |

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
- Are failed or filtered designs included in the denominator?
- Is the validation model independent from the generator's training signal?
- Does the design claim depend on sequence, structure, complex, or experimental evidence?

## Related

- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[molecular-modeling/protein-modeling|Protein modeling]]
- [[molecular-modeling/paper-claim-patterns|Computational Biology paper claim patterns]]
- [[concepts/ai-computational-biology-math-contract|AI Computational Biology Math contract]]
