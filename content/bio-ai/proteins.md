---
title: Proteins
tags:
  - bio-ai
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

## Checks

- Are homologs and protein families separated across train/test?
- Are residue indexing, missing residues, mutations, and chain choices explicit?
- Is the structure source experimental, predicted, apo, holo, or complex?
- Does the model use templates, MSAs, or bound ligands that change the task?

## Related

- [[bio-ai/index|Bio-AI]]
- [[bio-ai/structure-based-ai|Structure-based AI]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
