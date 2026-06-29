---
title: Computational Biology Boundary
aliases:
  - computational-biology/boundary
  - bio/computational-biology
tags:
  - computational-biology
unlisted: true
---

# Computational Biology Boundary

Computational Biology in this site means the object and workflow side of modeling biological and chemical systems. It covers molecular, protein, structure, interaction, and sequence-level modeling, but it does not try to survey all of biology. Broad omics, systems biology, transcriptomics, single-cell analysis, and clinical biology stay out of scope unless a concrete research need appears.

The practical unit is:

$$
\text{biological object}
+
\text{measurement context}
+
\text{computational representation}
\rightarrow
\text{claim}
$$

If the object, measurement, representation, or claim is vague, the model result is hard to interpret.

## Scope

| Area | Use For |
| --- | --- |
| [Computational Biology Scope](/molecular-modeling/modeling-scope) | route choice and boundary between domain object, AI method, Math, and Agents |
| [Objects and Entities](/molecular-modeling/entities) | protein, molecule, ligand, pocket, complex, assay, sequence, structure |
| [Sequence-based modeling](/molecular-modeling/sequence-based) | protein sequence, genome sequence, representation, family or region split |
| [Molecular and ligand modeling](/molecular-modeling/molecular-ligand) | small-molecule identity, standardization, fingerprints, graphs, conformers |
| [Interaction modeling](/molecular-modeling/interactions) | target-conditioned activity, affinity, selectivity, pair-level claims |
| [Structure-based modeling](/molecular-modeling/structure-based) | protein-ligand geometry, pose, interaction, scoring |
| [Docking](/molecular-modeling/docking) | receptor/ligand preparation, pose generation, pose quality, ranking |
| [Data and evaluation](/molecular-modeling/data-evaluation) | split units, label semantics, leakage, benchmark contracts |

## Core Questions

- What is the biological or chemical object?
- What is the context: target, pocket, assay, condition, organism, or source?
- What representation is used: string, graph, fingerprint, sequence embedding, coordinate set, or complex graph?
- What split unit supports the claim: scaffold, protein family, complex pair, assay, or time?
- What metric answers the actual task rather than a convenient proxy?

## Representation Map

| Object | Common Representations | Main Risk |
| --- | --- | --- |
| Molecule / ligand | SMILES, molecular graph, fingerprint, conformer ensemble | identity drift from salt, tautomer, protonation, stereo, or standardization choices |
| Protein | amino-acid sequence, MSA, structure, residue graph, pocket representation | family leakage, residue-index mismatch, missing regions, structure-cleaning artifacts |
| Protein-ligand complex | pocket-ligand coordinates, interaction graph, distance/contact features | template leakage, ligand-defined pocket leakage, symmetry-incorrect pose comparison |
| Assay label | endpoint, unit, threshold, censoring, source, target context | mixing incompatible assays or treating weak labels as clean labels |
| Genome region | sequence window, k-mer counts, variant context, annotation features | coordinate-system mismatch and overbroad omics claims |

## Claim Template

A public computational biology claim should be reducible to:

$$
(\text{object},\ \text{context},\ \text{representation},\ \text{split},\ \text{metric})
\rightarrow
\text{claim}
$$

For example, a docking note should not only say that a model performs well. It should say whether the claim is about pose generation, pose plausibility, ranking, affinity prediction, enrichment, or out-of-family generalization.

## Boundaries

| Include | Defer |
| --- | --- |
| structure-based modeling | broad systems biology |
| molecular and ligand modeling | transcriptomics and single-cell analysis |
| sequence-based protein/genome modeling | clinical omics |
| interaction modeling | pathway biology unless a concrete project needs it |
| assay-aware labels and splits | private datasets or unpublished results |

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/modeling-scope|Computational Biology Scope]]
- [[concepts/data/dataset-construction-checklist|Dataset construction checklist]]
- [[concepts/evaluation/leakage|Leakage]]
- [[entities/entity-relation-map|Entity relation map]]
- [[concepts/molecular-modeling/index|Molecular modeling concepts]]
- [[concepts/protein-modeling/index|Protein modeling concepts]]
