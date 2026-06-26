---
title: Protein
tags:
  - entities
  - protein
---

# Protein

Proteins are biological macromolecules represented as sequences, structures, surfaces, graphs, or dynamics depending on the task.

A protein representation can be sequence-only, structure-only, or fused:

$$
r_P = \phi(s_{1:L}, X, c)
$$

where $s_{1:L}$ is the residue sequence, $X$ is optional structure or coordinate information, and $c$ is context such as domain, family, pocket, mutation, construct, or assay.

## Modeling Views

- Sequence view for language-model and [[concepts/learning/self-supervised-learning|self-supervised learning]] methods.
- Structure view for folding, docking, and [[concepts/geometric-deep-learning/equivariant-gnn|equivariant]] models.
- Functional view for binding sites, domains, and interactions.
- Target view for assays, docking benchmarks, and virtual screening.
- Dataset view for homolog grouping, sequence identity clustering, and protein-family splits.

## Identity Boundary

A protein record is not only a sequence string. Public notes should distinguish:

$$
P
=
(\text{accession},
\text{isoform},
\text{construct},
\text{chain},
\text{mutation state},
\text{sequence},
\text{structure source})
$$

The same biological target can appear as a full-length sequence, a domain construct, a mutant, a crystal chain, a predicted model, or an assay-specific construct. These should not be silently collapsed unless the task explicitly allows it.

## Structure Availability

For modeling, the most important question is whether structure is available at deployment time:

$$
\text{input availability}
\in
\{\text{sequence only},
\text{predicted structure},
\text{experimental apo},
\text{experimental holo},
\text{known complex}\}
$$

Using a holo structure or ligand-defined pocket during training can be valid for analysis, but it changes the task if the deployment setting only has sequence or apo structure.

## Split and Leakage

Protein-side generalization usually needs sequence or family grouping:

$$
\operatorname{cluster}(P_i)
\ne
\operatorname{cluster}(P_j)
\quad
\text{for train/test separation}
$$

where the cluster can be based on sequence identity, domain family, fold, or target class. For structure-based tasks, also record whether homolog templates or known complexes were available to the model pipeline.

## Checks

- Is the task sequence-only, structure-only, or sequence-structure combined?
- Are homologs, protein families, and near-duplicate chains separated across splits?
- Are domains, missing residues, mutations, and cofactors represented explicitly?
- Does the model predict structure, function, interaction, or a downstream assay label?
- Is the split grouped by protein family, sequence identity, target, or assay?
- Is the structure experimental, predicted, modeled, or unavailable at deployment time?
- Does the representation include information that would not be available for the intended task?
- Are isoform, construct, mutation, and chain choices recorded?
- Are sequence embeddings aligned to the same residue indexing as structure coordinates?
- Are templates, MSAs, and bound ligands treated as part of the evidence boundary?

## Related

- [[entities/entity-relation-map|Entity relation map]]
- [[entities/pocket|Pocket]]
- [[entities/assay|Assay]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/protein-modeling/sequence-identity-clustering|Sequence identity clustering]]
- [[concepts/protein-modeling/protein-domain|Protein domain]]
- [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]]
- [[concepts/protein-modeling/sequence-structure-alignment|Sequence-structure alignment]]
- [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[research/protein-modeling/index|Protein modeling]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[research/protein-modeling/mambafold|MambaFold]]
