---
title: Objects and Entities
aliases:
  - computational-biology/entities
  - bio/entities
tags:
  - computational-biology
  - entities
---


# Objects and Entities

Computational biology에서 먼저 정해야 하는 것은 모델이 다루는 대상입니다. 같은 단어라도 protein, ligand, target, assay, structure가 어떤 단위로 정의되는지에 따라 split, leakage, evaluation이 달라집니다.

$$
x_{\mathrm{bio}}
\in
\{\text{protein}, \text{ligand}, \text{pocket}, \text{complex}, \text{assay}, \text{genome region}\}
$$

## Core Objects

| Object | Use When | Main Risk |
| --- | --- | --- |
| [[entities/target|Target]] | target-conditioned activity, affinity, selectivity, assay context | target identity may hide isoform, construct, mutation, species, or assay source |
| [[entities/protein|Protein]] | sequence, structure, representation, family split | homolog leakage and residue-index mismatch |
| [[entities/pocket|Pocket]] | binding site, docking, pose prediction, structure-based generation | pocket may be ligand-defined or unavailable at inference |
| [[entities/ligand|Ligand]] | target-bound molecule, pose, activity, interaction | ligand state and atom mapping may differ across protocols |
| [[entities/molecule|Molecule]] | property prediction, molecular generation, similarity, scaffold split | salt, stereo, tautomer, protonation, and duplicate handling |
| [[entities/protein-ligand-complex|Protein-ligand complex]] | docking, pose quality, interaction prediction, complex graph | pair-level split can leak protein or ligand families |
| [[entities/sequence|Sequence]] | protein language models, genome region models, motif tasks | sequence context and split unit may not match the biological claim |
| [[entities/structure|Structure]] | coordinates, conformers, protein structures, complexes | template, frame, and coordinate-source leakage |
| [[entities/genome|Genome]] | sequence/region/variant-level modeling only | broad omics claims are out of scope unless explicitly added |

Use [[entities/entity-relation-map|Entity relation map]] when a note has several objects and the relation matters.

## Entity Tuple

For a paper note, write the modeled unit as a tuple rather than a single word:

$$
u
=
(\text{entity}, \text{context}, \text{measurement}, \text{split group})
$$

For example, a row may be molecule-only for property prediction, but target-conditioned activity needs molecule, target, assay, endpoint, unit, and source. A protein-ligand pose needs protein structure, pocket rule, ligand state, atom mapping, pose coordinates, and pose source.

## Split Implication

| Claim | Hold Out |
| --- | --- |
| new molecules generalize | scaffold, chemical series, or time split |
| new proteins generalize | sequence identity, family, fold, or target split |
| new protein-ligand pairs generalize | both ligand and protein axes, or an explicit pair-level claim |
| new structures generalize | template-aware, homolog-aware, or coordinate-source-aware split |
| new assays generalize | assay/source split or endpoint harmonization check |
| new genome regions generalize | chromosome, locus, family, or annotation-source split |

## Label Objects

- [[entities/target-assay-label|Target-assay-label contract]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]

## Checks

- Is the entity a biological object, chemical object, assay record, or derived feature?
- Are isoform, construct, mutation, chain, ligand state, and assay context explicit?
- Is one row an object, a pair, a pose, a measurement, a generated sample, or a candidate list?
- Are target, assay, endpoint, unit, threshold, censoring, and source preserved when labels are involved?
- Does the representation use a ligand-defined pocket, known pose, template, homolog, or future annotation unavailable at inference?
- Is the split unit the same as the biological claim?
- Does the input include information unavailable at deployment time?

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/sequence-based|Sequence-based modeling]]
- [[molecular-modeling/molecular-ligand|Molecular and ligand modeling]]
- [[molecular-modeling/interactions|Interaction modeling]]
- [[molecular-modeling/data-evaluation|Data and evaluation]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[concepts/modalities/representation-contract|Representation contract]]
- [[concepts/evaluation/leakage|Leakage]]
