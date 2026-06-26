---
title: Protein-Ligand Complex
tags:
  - entities
  - protein-ligand
  - docking
---

# Protein-Ligand Complex

A protein-ligand complex combines a protein binding site, ligand conformation, and interaction geometry. It is a central object in structure-based modeling.

A complex can be viewed as:

$$
C = (P, L, X_P, X_L, I)
$$

where $P$ is the protein or pocket, $L$ is the ligand identity, $X_P$ and $X_L$ are coordinates, and $I$ is the interaction context used by a model or evaluator.

## Questions

- Is the pose geometrically plausible?
- Does the [[concepts/sbdd/scoring-function|scoring function]] measure pose quality, binding affinity, or both?
- Which representation best preserves geometry and chemistry?
- Is the task pose generation, pose ranking, affinity prediction, or virtual screening?

## Example Unit

A complex example is usually a pair plus geometry:

$$
e
=
(P, L, X_P, X_L, y, s)
$$

where $y$ is an optional label and $s$ is the source or protocol. The label may describe pose quality, assay activity, binding affinity, or benchmark membership. These should not be mixed without naming the target.

## Geometry and Label Boundary

For docking and SBDD, geometry and assay evidence are different:

$$
\text{pose quality}
\ne
\text{binding affinity}
\ne
\text{virtual-screening rank}
$$

A pose can be geometrically plausible but biologically weak. A ligand can be active in an assay while a generated pose is wrong. Notes should state which claim is being evaluated.

## Split Policy

Complex-level splits need both sides:

$$
\operatorname{split\_key}(C)
=
(\operatorname{protein\_family}(P),
\operatorname{ligand\_scaffold}(L),
\operatorname{source}(C))
$$

For stronger claims, hold out both protein families and ligand scaffolds. Holding out only row IDs is usually too weak for structure-based generalization.

## Checks

- Is the binding site defined from a structure, pocket detector, or known ligand?
- Are clashes, bond geometry, chirality, and protein-ligand contacts validated?
- Does the model condition on the full protein, local pocket, surface, or residue graph?
- Are pose quality and binding affinity evaluated as separate targets?
- Are ligand scaffolds and protein families both controlled in the split?
- Is the complex experimental, docked, predicted, generated, or used only as a reference?
- Does the label come from the complex geometry, an assay, or a curated benchmark?
- Is the protein structure apo, holo, predicted, or transferred from a template?
- Are water, metals, cofactors, and protonation states included or removed by protocol?
- Are invalid poses counted as failures rather than filtered after scoring?

## Related

- [[entities/entity-relation-map|Entity relation map]]
- [[entities/protein|Protein]]
- [[entities/pocket|Pocket]]
- [[entities/ligand|Ligand]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/sbdd/interaction-fingerprint|Interaction fingerprint]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/sbdd/docking-workflow|Docking workflow]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]
