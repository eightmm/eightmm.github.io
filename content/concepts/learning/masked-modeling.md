---
title: Masked Modeling
tags:
  - masked-modeling
  - self-supervised-learning
  - representation-learning
---

# Masked Modeling

Masked modeling hides part of an input and trains the model to predict the missing content from the visible context. The masked target can be tokens, patches, nodes, or residues.

The typical objective is:

$$
\mathcal{L}
= -\sum_{i\in M}
\log p_\theta(x_i \mid x_{\setminus M})
$$

Here $M$ is the masked subset and $x_{\setminus M}$ is the visible context.

For continuous targets, the reconstruction loss may be:

$$
\mathcal{L}
=
\sum_{i\in M}
\lVert \hat{x}_i - x_i\rVert_2^2
$$

where $\hat{x}_i$ is the model's prediction for the masked element.

## Mask Sampling Contract

The masked set $M$ is part of the objective:

$$
M \sim q(M\mid x)
$$

Different sampling rules create different learning problems.

| Mask Rule | What It Tests | Risk |
| --- | --- | --- |
| independent random mask | local redundancy and context use | task may be too easy |
| span mask | longer-range dependency | target may become ambiguous |
| block or patch mask | spatial context | boundary artifacts |
| node/edge mask | graph context | graph construction can leak target |
| residue/atom mask | biological or chemical context | local chemistry may dominate |
| coordinate/distance mask | geometric consistency | alignment or template leakage |

## Discrete and Continuous Targets

For discrete targets, the objective is usually cross-entropy over masked positions:

$$
\mathcal{L}_{\mathrm{disc}}
=
-\sum_{i\in M}
\log p_\theta(x_i\mid x_{\setminus M})
$$

For continuous targets, the loss defines which errors matter:

$$
\mathcal{L}_{\mathrm{cont}}
=
\sum_{i\in M}
\rho(\hat{x}_i,x_i)
$$

where $\rho$ may be squared error, absolute error, negative log-likelihood, or a geometry-aware distance.

## Masking Design

- Random token or patch masking tests local context recovery.
- Span masking forces longer-range reasoning.
- Node or edge masking tests graph context.
- Residue or atom masking tests biological or chemical context.
- Coordinate or distance masking tests geometric consistency.

## Shortcut Risks

Masked modeling can fail as representation learning even when pretraining loss is low.

| Shortcut | Example |
| --- | --- |
| local copy | neighboring tokens or atoms reveal the answer |
| metadata leakage | mask indicator, position, or template source reveals target |
| distribution shortcut | frequent token/class dominates prediction |
| reconstruction overfit | model learns nuisance detail not useful downstream |
| split leakage | overlapping windows or homologs cross pretrain/eval boundary |

The downstream claim should be checked with [[concepts/learning/representation-evaluation|representation evaluation]], not only masked loss.

## Why It Matters

- A simple, scalable self-supervised objective across text, images, graphs, and sequences.
- Builds context-aware representations without manual labels.
- Underlies many pretrained encoders for language, vision, and biomolecules.
- The masking pattern determines whether the task is trivial, useful, or impossible.

## Checks

- Does the masking ratio and pattern match the data's redundancy?
- Is the prediction target reconstructed in pixel/token space or in a latent space?
- Could trivial shortcuts solve the pretext task without learning structure?
- Are masked positions excluded from the visible input and metadata?
- Does the objective learn reusable representations, or only local reconstruction tricks?
- Is the masked loss discrete, continuous, probabilistic, or geometry-aware?
- Are masking rules fit to the modality and downstream claim?
- Are overlapping windows, homologs, scaffolds, or templates handled before evaluation?

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/augmentation-policy|Augmentation policy]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/machine-learning/cross-entropy-loss|Cross-entropy loss]]
- [[concepts/machine-learning/mean-squared-error|Mean squared error]]
- [[concepts/evaluation/test-set-contamination|Test set contamination]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
