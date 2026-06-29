---
title: Pocket Representation
tags:
  - protein-modeling
  - sbdd
  - representation-learning
---

# Pocket Representation

Pocket representation describes how a protein binding site is converted into model input. The pocket can be represented as residues, atoms, surfaces, grids, graphs, coordinates, or learned embeddings.

A distance-defined pocket around ligand coordinates $X_L$ can be:

$$
\mathcal{P}_r
=
\{i \in P : \min_{j\in L}\lVert x_i-x_j\rVert_2 \le r\}
$$

where $r$ is a cutoff radius.

The pocket representation is then:

$$
z_{\mathcal{P}}
=
f_\theta(\mathcal{P}_r, X_{\mathcal{P}}, A_{\mathcal{P}})
$$

where $X_{\mathcal{P}}$ are coordinates or features and $A_{\mathcal{P}}$ is optional adjacency or contact structure.

## Common Representations

| Representation | Captures | Main risk |
| --- | --- | --- |
| residue sequence window | local sequence context | misses 3D proximity and long-range contacts |
| residue graph/contact map | residue-level geometry | cutoff and missing residues change graph |
| atom-level graph | local chemistry and coordinates | atom typing, protonation, and side-chain uncertainty |
| 3D grid or voxel | spatial density around pocket | resolution, frame, and memory cost |
| surface patch | shape and chemical surface features | surface generation parameters affect features |
| learned embedding | compact representation for retrieval or prediction | hard to audit what information is preserved |

## Pocket Source Contract

| Source | Available at deployment? | Claim boundary |
| --- | --- | --- |
| known binding site | yes, if site is annotated | tests site-conditioned modeling, not blind site discovery |
| predicted binding site | yes, if predictor is part of workflow | includes predictor error in the pipeline |
| ligand-defined pocket | no for unknown binders | useful for retrospective analysis, risky for blind prediction |
| residue cutoff from reference complex | no for new pairs | can leak the answer if used in evaluation |
| whole-protein input | yes | shifts burden to model to find relevant region |

If a pocket is defined from a bound test ligand, the task is no longer blind pocket discovery or deployment-like docking.

## Key Risks

- Pocket extraction can leak ligand information if defined from the test ligand.
- Different cutoffs can change the apparent task.
- Missing side chains, cofactors, metals, and waters may alter interactions.
- Whole-protein context can matter even when the model only sees a local pocket.
- Pocket similarity across train and test can hide target-family leakage.
- Predicted pockets can fail silently if the benchmark only scores successful extractions.

## Checks

- Is the pocket known, predicted, transferred, or ligand-defined?
- What atoms, residues, chains, waters, metals, and cofactors are included?
- Does the representation preserve rotation/translation invariance or equivariance?
- Is the pocket representation available at deployment time?
- Is pocket similarity controlled across train and test?
- Are failed pocket detections, empty pockets, and missing structures counted?
- Does the evaluation separate pocket detection quality from downstream scoring quality?

## Related

- [[entities/pocket|Pocket]]
- [[concepts/protein-modeling/binding-site|Binding site]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
