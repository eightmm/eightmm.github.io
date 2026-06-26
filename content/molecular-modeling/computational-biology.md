---
title: Computational Biology
aliases:
  - bio/computational-biology
tags:
  - molecular-modeling
  - computational-biology
---

# Computational Biology

Computational biology in this site is a focused support layer for AI-driven molecular, protein, structure, and sequence modeling. It is not a broad survey of omics, systems biology, or clinical biology.

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
| [Molecular Modeling Scope](/molecular-modeling/modeling-scope) | naming, route choice, and boundary between domain, AI method, Math, and Agents |
| [Entities](/molecular-modeling/entities) | protein, molecule, ligand, pocket, complex, assay, sequence, structure |
| [Molecules](/molecular-modeling/molecules) | small-molecule identity, standardization, fingerprints, graphs, conformers |
| [Proteins](/molecular-modeling/proteins) | sequence, structure, domains, binding sites, representations |
| [Structure-based modeling](/molecular-modeling/structure-based) | protein-ligand geometry, pose, interaction, scoring |
| [Docking](/molecular-modeling/docking) | receptor/ligand preparation, pose generation, pose quality, ranking |
| [Data and evaluation](/molecular-modeling/data-evaluation) | split units, label semantics, leakage, benchmark contracts |
| [Genome](/molecular-modeling/genome) | sequence, region, k-mer, variant-effect modeling |

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

A public Molecular Modeling claim should be reducible to:

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
| molecular modeling | transcriptomics and single-cell analysis |
| protein modeling | clinical omics |
| genome sequence / variant-level modeling | pathway biology unless a concrete project needs it |
| assay-aware labels and splits | private datasets or unpublished results |

## Related

- [[molecular-modeling/index|Molecular Modeling]]
- [[molecular-modeling/modeling-scope|Molecular Modeling Scope]]
- [[concepts/data/dataset-construction-checklist|Dataset construction checklist]]
- [[concepts/evaluation/leakage|Leakage]]
- [[entities/entity-relation-map|Entity relation map]]
- [[concepts/molecular-modeling/index|Molecular modeling concepts]]
- [[concepts/protein-modeling/index|Protein modeling concepts]]
