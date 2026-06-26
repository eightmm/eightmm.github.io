---
title: Proteins
aliases:
  - bio/proteins
  - bio-ai/proteins
tags:
  - bio
  - proteins
---


# Proteins

Protein modeling covers sequence, structure, domains, binding sites, and learned representations. The important distinction is whether the model sees sequence only, predicted structure, experimental structure, or a known complex.

$$
r_P = \phi(s_{1:L}, X, c)
$$

where $s_{1:L}$ is the residue sequence, $X$ is optional coordinate information, and $c$ is context such as family, domain, pocket, mutation, or assay condition.

## Core Notes

- [[entities/protein|Protein]]
- [[entities/sequence|Sequence]]
- [[entities/structure|Structure]]
- [[concepts/protein-modeling/index|Protein modeling concepts]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/protein-modeling/protein-domain|Protein domain]]
- [[concepts/protein-modeling/protein-structure-cleaning|Protein structure cleaning]]
- [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]]

## Representation Choices

| Representation | Use For | Main Risk |
| --- | --- | --- |
| Raw sequence | language-model pretraining, classification, mutation effect prediction | homolog leakage and truncation policy can dominate results |
| MSA / evolutionary profile | structure prediction, family-aware representation | MSA depth and template/database overlap can leak test information |
| Residue embedding | downstream supervised models and retrieval | pooling rule and special-token handling change the representation |
| Contact map / residue graph | structure-aware prediction without full coordinates | threshold choices and missing residues affect graph topology |
| 3D coordinates | pocket, docking, structure refinement, equivariant models | coordinate source, chain selection, alignment, and units must be explicit |

## Sequence to Structure Map

Many protein notes move through this chain:

$$
s_{1:L}
\rightarrow
h_{1:L}
\rightarrow
G_P\ \text{or}\ X_P
\rightarrow
\hat{y}
$$

Here $s_{1:L}$ is the amino-acid sequence, $h_{1:L}$ is a residue-level representation, $G_P$ is a residue/contact graph, $X_P$ is a coordinate set, and $\hat{y}$ is the task output.

## Sequence and Structure

- [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]]
- [[concepts/protein-modeling/sequence-identity-clustering|Sequence identity clustering]]
- [[concepts/protein-modeling/residue-indexing|Residue indexing]]
- [[concepts/protein-modeling/contact-map|Contact map]]
- [[concepts/protein-modeling/sequence-structure-alignment|Sequence-structure alignment]]

## Binding Context

- [[concepts/protein-modeling/binding-site|Binding site]]
- [[concepts/protein-modeling/pocket-representation|Pocket representation]]
- [[entities/pocket|Pocket]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]

## Claim Map

| Claim | Required Boundary |
| --- | --- |
| Sequence representation works | sequence identity split, pooling rule, model-selection protocol |
| Structure representation helps | source of structure, cleaning protocol, residue alignment, missing-region handling |
| Binding-site prediction works | pocket definition, ligand availability, apo/holo distinction, localization metric |
| Protein-ligand modeling generalizes | protein-family split plus ligand scaffold or complex-pair split |

## Checks

- Are homologs and protein families separated across train/test?
- Are residue indexing, missing residues, mutations, and chain choices explicit?
- Is the structure source experimental, predicted, apo, holo, or complex?
- Does the model use templates, MSAs, or bound ligands that change the task?

## Related

- [[molecular-modeling/index|Molecular Modeling]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
