---
title: Protein Modeling
aliases:
  - research/protein-modeling
tags:
  - molecular-modeling
  - protein-modeling
---

# Protein Modeling

Protein modeling notes cover folding, structure prediction, protein representation learning, and links to downstream structure-based tasks.

A broad protein modeling view is:

$$
\hat{X} = f_\theta(s, c)
$$

where $s$ is a protein sequence, $c$ is optional context, and $\hat{X}$ is a predicted structure or representation.

## Route Map

| Question | Start |
| --- | --- |
| What object is being modeled? | [[entities/protein|Protein]], [[entities/sequence|Sequence]], [[entities/structure|Structure]] |
| How is the protein represented? | [[concepts/protein-modeling/protein-representation|Protein representation]], [[concepts/modalities/sequence|Sequence]], [[concepts/modalities/3d-structure|3D structure]] |
| Does the method use evolutionary context? | [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]], [[concepts/protein-modeling/protein-domain|Protein domain]] |
| Does the method predict coordinates or contacts? | [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]], [[concepts/protein-modeling/contact-map|Contact map]], [[concepts/tasks/coordinate-prediction|Coordinate prediction]] |
| Is residue numbering or chain mapping important? | [[concepts/protein-modeling/residue-indexing|Residue indexing]], [[concepts/protein-modeling/sequence-structure-alignment|Sequence-structure alignment]] |
| Is this linked to binding or docking? | [[molecular-modeling/structure-based/index|Structure-based modeling]], [[entities/protein-ligand-complex|Protein-ligand complex]], [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]] |
| What split tests transfer? | [[concepts/evaluation/protein-family-split|Protein family split]], [[concepts/evaluation/ood-generalization|OOD generalization]] |

## Modeling Modes

| Mode | Typical Input | Typical Output | Evaluation Risk |
| --- | --- | --- | --- |
| sequence representation | sequence, MSA, domain annotation | embedding, residue features | homolog leakage, family imbalance |
| structure prediction | sequence, template, MSA | coordinates, distances, contacts | template leakage, atom/residue mapping |
| functional prediction | sequence or structure | class, site, activity, property | label semantics, assay/source leakage |
| interaction modeling | protein plus ligand, peptide, protein, or nucleic acid | binding score, pose, contact, interface | split must control both interaction sides |
| generative design | sequence, backbone, pocket, constraints | sequence, structure, complex, design candidates | validity, novelty, diversity, downstream assay gap |

## Evidence Fields

| Field | Why It Matters |
| --- | --- |
| sequence source | duplicate proteins and homologs define leakage risk |
| structure source | experimental, predicted, template-derived, and relaxed structures support different claims |
| representation unit | token, residue, domain, chain, complex, pocket, or full structure changes the task |
| split unit | row split, protein family split, fold split, target split, and complex split test different generalization claims |
| metric | sequence, coordinate, ranking, binding, and generation metrics are not interchangeable |
| uncertainty | replicate noise, structural ambiguity, seed variance, and family imbalance affect interpretation |

## Adjacent Areas

- [[concepts/protein-modeling/index|Protein modeling concepts]]
- [[concepts/protein-modeling/protein-structure-cleaning|Protein structure cleaning]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
- [[ai/generative-models|Generative models]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[papers/protein-modeling/meet-equivariant-peptide|MEET]]
