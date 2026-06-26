---
title: Entity Relation Map
tags:
  - entities
  - molecular-modeling
  - structure-based-modeling
---

# Entity Relation Map

Entity notes define what object a model is operating on. In Molecular Modeling, the same model family can mean different things depending on whether the unit is a molecule, protein, pocket, complex, assay row, or dataset.

The central relation for structure-based modeling is:

$$
\text{protein} + \text{pocket} + \text{ligand}
\rightarrow
\text{protein-ligand complex}
$$

The central relation for supervised chem-bio ML is:

$$
(\text{molecule}, \text{target}, \text{assay})
\rightarrow
\text{bioactivity label}
$$

This is the [[entities/target-assay-label|Target-assay-label contract]].

## Object Hierarchy

- [[entities/molecule|Molecule]] is the general chemical object.
- [[entities/ligand|Ligand]] is a molecule considered as a binding partner.
- [[entities/target|Target]] is the biological object a task is centered on.
- [[entities/protein|Protein]] is the biological target or modeled macromolecule.
- [[entities/pocket|Pocket]] is a local region of a protein.
- [[entities/protein-ligand-complex|Protein-ligand complex]] combines ligand pose and protein context.
- [[entities/assay|Assay]] defines how a label was measured.
- [[entities/bioactivity-label|Bioactivity label]] records the measured or curated response.
- [[entities/dataset|Dataset]] collects examples, labels, metadata, and splits.
- [[entities/target-assay-label|Target-assay-label contract]] connects the measured object, measurement protocol, label semantics, and split rule.

## Representation Choices

- Sequence: residues, nucleotides, SMILES tokens, or k-mers.
- Graph: atoms, residues, bonds, contacts, or interaction edges.
- Structure: 3D coordinates, distances, local frames, surfaces, or voxels.
- Metadata: assay protocol, source, target, assembly, or split group.

## Split Units

The split unit should match the generalization claim:

- Molecule-side generalization: scaffold or molecular cluster.
- Protein-side generalization: sequence or structure family.
- Complex-side generalization: both ligand scaffold and protein family.
- Assay-side generalization: assay, campaign, target, or source group.
- Genome-side generalization: region, chromosome, species, or source group.

## Checks

- What is one example?
- What is the target label?
- What target, assay, and context produced the label?
- Is the label assay-conditioned, target-conditioned, censored, transformed, or thresholded?
- Which entities are inputs and which are metadata?
- Which entity defines the split group?
- Could the same molecule, homologous protein, related complex, or shared assay leak across splits?
- Does the representation preserve the chemistry or geometry needed for the task?

## Related

- [[molecular-modeling/index|Molecular Modeling]]
- [[entities/target|Target]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[concepts/data/index|Data]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/sbdd/index|Structure-based drug discovery]]
