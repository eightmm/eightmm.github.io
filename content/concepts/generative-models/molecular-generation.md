---
title: Molecular Generation
tags:
  - molecular-generation
  - generative-model
  - drug-discovery
---

# Molecular Generation

Molecular generation produces novel molecular structures—as strings, graphs, or 3D coordinates—often under constraints such as validity, synthesizability, or target properties.

Conditional molecular generation can be written as:

$$
m \sim p_\theta(m \mid c)
$$

where $m$ is a molecule and $c$ can represent a target property, scaffold, protein pocket, or design constraint.

## Representation Choices

Molecules can be generated in several spaces:

- SMILES or SELFIES strings.
- Molecular graphs.
- Fragments or reaction templates.
- 3D conformers or protein-ligand poses.
- Latent variables decoded into molecules.

Each representation changes the constraint problem. A string model must learn syntax. A graph model must enforce valence and connectivity. A 3D model must respect geometry, chirality, and conformational uncertainty.

## Objective View

A property-conditioned generator often seeks samples that balance likelihood and utility:

$$
m^\*
=
\arg\max_m
\left[
\log p_\theta(m\mid c)
+
\lambda U(m)
-
\gamma C(m)
\right]
$$

where $U(m)$ is a utility such as predicted activity and $C(m)$ is a penalty for invalidity, synthetic difficulty, toxicity proxy, or violating design constraints.

For structure-based generation, conditioning may include a pocket representation:

$$
m
\sim
p_\theta(m\mid P, c)
$$

where $P$ is a protein pocket or receptor context. This makes split design and template leakage especially important.

## Why It Matters

- A core tool for de novo design in drug discovery and materials.
- Spans autoregressive, VAE, flow, and diffusion approaches.
- Useful output requires validity and property control, not just novelty.
- Optimization against learned predictors can exploit predictor artifacts.

## Evaluation

Generated molecules should be evaluated after standardization:

$$
m
\xrightarrow{\text{standardize}}
\tilde{m}
\xrightarrow{\text{dedup}}
\text{validity/novelty/property checks}
$$

Common metrics include validity, uniqueness, novelty, diversity, property distribution, scaffold distribution, synthesizability proxies, and domain-specific hit-rate. For structure-based generation, pose quality, steric clashes, interaction plausibility, and receptor split policy also matter.

Validity alone is weak. A generator that produces many valid molecules can still fail by memorizing training scaffolds, exploiting a biased scoring model, or generating molecules outside the applicability domain of the property predictor.

## Checks

- Are generated molecules valid, novel, and synthesizable?
- Does the representation (SMILES, graph, 3D) suit the objective?
- Is property conditioning evaluated against held-out targets?
- Are molecules standardized before duplicate and novelty checks?
- Is novelty computed against the train set, known databases, or both?
- Is the property model calibrated and in-domain for generated samples?
- Are ligand scaffolds, protein families, and templates separated when claiming structure-based generalization?
- Are decoy-like shortcuts or trivial property biases ruled out with cheap baselines?

## Related

- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[concepts/sbdd/template-leakage|Template leakage]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/protein-design|Protein design]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
