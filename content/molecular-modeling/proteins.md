---
title: Proteins
aliases:
  - computational-biology/proteins
  - bio/proteins
tags:
  - computational-biology
  - proteins
---


# Proteins

Protein modeling covers sequence, structure, domains, binding sites, and learned representations. The important distinction is whether the model sees sequence only, predicted structure, experimental structure, or a known complex.

$$
r_P = \phi(s_{1:L}, X, c)
$$

where $s_{1:L}$ is the residue sequence, $X$ is optional coordinate information, and $c$ is context such as family, domain, pocket, mutation, or assay condition.

## Route Map

| Question | Start | Watch |
| --- | --- | --- |
| What is the modeled object? | [Protein](/entities/protein), [Sequence](/entities/sequence), [Structure](/entities/structure) | chain choice, isoform, construct, mutation, missing residues |
| Is the input sequence-only or structure-aware? | [Protein representation](/concepts/protein-modeling/protein-representation), [Protein structure prediction](/concepts/protein-modeling/protein-structure-prediction) | using predicted or template-derived structure as if it were available in deployment |
| Which biological unit is preserved? | [Protein domain](/concepts/protein-modeling/protein-domain), [Sequence identity clustering](/concepts/protein-modeling/sequence-identity-clustering) | homolog leakage and domain truncation |
| Is structure preprocessing part of the method? | [Protein structure cleaning](/concepts/protein-modeling/protein-structure-cleaning), [Residue indexing](/concepts/protein-modeling/residue-indexing) | silent residue renumbering, missing atoms, chain filtering |
| Is the task about binding context? | [Binding site](/concepts/protein-modeling/binding-site), [Pocket representation](/concepts/protein-modeling/pocket-representation), [Protein-ligand complex](/entities/protein-ligand-complex) | apo/holo distinction and ligand-defined pockets |

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

## Sequence and Structure Routes

| Area | Start | Use For |
| --- | --- | --- |
| Evolutionary context | [Multiple sequence alignment](/concepts/protein-modeling/multiple-sequence-alignment), [Sequence identity clustering](/concepts/protein-modeling/sequence-identity-clustering) | homolog control, family split, MSA-dependent methods |
| Structure graph | [Contact map](/concepts/protein-modeling/contact-map), [Sequence-structure alignment](/concepts/protein-modeling/sequence-structure-alignment) | residue graph construction and coordinate-aware representations |
| Binding context | [Binding site](/concepts/protein-modeling/binding-site), [Pocket representation](/concepts/protein-modeling/pocket-representation), [Pocket](/entities/pocket) | pocket-level prediction, docking, protein-ligand interaction |

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

- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/interactions|Interaction modeling]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
