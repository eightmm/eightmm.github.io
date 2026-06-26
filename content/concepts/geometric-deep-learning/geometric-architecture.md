---
title: Geometric Architecture
tags:
  - geometric-deep-learning
  - architectures
---

# Geometric Architecture

A geometric architecture builds symmetry assumptions directly into a model. It is used when inputs include coordinates, directions, graphs, or physical structures.

The core design target is:

$$
F(g\cdot x) = \rho(g)F(x)
$$

where $g$ is a transformation such as rotation or translation, and $\rho(g)$ specifies how the output should transform.

The design starts by matching input type, target type, and group:

$$
(\text{input object}, \text{target}, G)
\rightarrow
(\text{features}, \text{messages}, \text{readout})
$$

For scalar prediction under rigid motions:

$$
F(RX+\mathbf{1}t^\top)=F(X)
$$

For coordinate prediction:

$$
F(RX+\mathbf{1}t^\top)
=
RF(X)+\mathbf{1}t^\top
$$

These are different architecture contracts.

## Design Choices

- Use invariant features such as distances for scalar outputs.
- Use equivariant coordinate or vector updates for structure prediction.
- Keep scalar, vector, and higher-order channels separate when needed.
- Use graph construction that respects the intended geometry.
- Match the final readout to the target: scalar score, vector field, coordinates, or distribution.

## Architecture Patterns

- Distance-based GNN: invariant edge features, good for scalar readouts.
- Equivariant coordinate GNN: scalar messages plus vector coordinate updates.
- Tensor/irrep network: higher-order features that transform under SO(3) representations.
- Local-frame model: expresses geometry in learned or constructed local frames.
- Hybrid model: invariant encoder plus equivariant refinement or generative head.

## Data Contract

- Coordinates should state source: experimental, predicted, docked, generated, simulated, or conformer-generated.
- Molecular inputs should state stereochemistry, protonation, tautomer, and conformer choices when relevant.
- Protein structures should state chain selection, missing residues, alternate locations, and residue alignment when relevant.
- Splits should match the generalization claim: ligand scaffold, protein family, complex pair, assay/source, or time.

## Failure Modes

- The architecture is symmetric but graph construction uses leakage-prone context.
- The target is scalar but preprocessing centers on a known ligand pose unavailable at deployment.
- Reflection symmetry is enforced on chirality-sensitive chemistry.
- Batch-level pooling mixes multiple molecules or complexes.

## Checks

- Is the target invariant or equivariant?
- Should reflections be included, or does chirality matter?
- Does preprocessing choose an arbitrary frame?
- Are generated coordinates evaluated for physical plausibility?
- Is the split unit appropriate for the claimed generalization?
- Are train and deployment coordinate sources compatible?

## Related

- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[concepts/geometric-deep-learning/distance-geometry|Distance geometry]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/sbdd/template-leakage|Template leakage]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
