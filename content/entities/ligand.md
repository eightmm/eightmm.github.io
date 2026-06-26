---
title: Ligand
tags:
  - entities
  - ligand
  - small-molecule
---

# Ligand

A ligand is a small molecule or molecular fragment considered as a binding partner for a protein target.

A ligand is not only a molecule identity. In structure-based tasks it also has a pose, conformation, protonation state, and pocket context:

$$
\ell = (m, X_L, q, P)
$$

where $m$ is molecular identity, $X_L$ is ligand coordinates, $q$ represents chemical state and preparation choices, and $P$ is the protein or pocket context.

## Modeling Views

- Graph view for atoms, bonds, and molecular topology.
- Conformer view for 3D geometry and pose generation.
- Interaction view for [[research/structure-based-ai/protein-ligand-docking|protein-ligand docking]].
- Dataset view for scaffold splits, [[entities/bioactivity-label|bioactivity labels]], and assay provenance.

## Identity vs State

Ligand identity and ligand state should be separated:

$$
L
=
(\text{standardized molecule},
\text{stereochemistry},
\text{tautomer},
\text{protonation},
\text{conformer},
\text{pose})
$$

The same raw ligand string can map to different model inputs after salt stripping, tautomer normalization, protonation, chirality handling, or conformer generation. Deduplication and splitting should use the standardized identity that matches the intended task.

## Pose Boundary

For structure-based tasks, a ligand can be:

$$
\text{pose source}
\in
\{\text{experimental},
\text{docked},
\text{generated},
\text{transferred},
\text{reference only}\}
$$

Pose source matters because a model trained with experimental poses may not be solving the same problem as a model that must generate poses from 2D molecules and a pocket.

## Split and Assay Context

Ligand-side generalization should control analog leakage:

$$
\operatorname{scaffold}(L_{\mathrm{train}})
\cap
\operatorname{scaffold}(L_{\mathrm{test}})
=
\varnothing
$$

when the claim is scaffold generalization. For activity labels, a ligand without target and assay context is not enough; use the [[entities/target-assay-label|target-assay-label contract]].

## Checks

- Is stereochemistry represented and preserved?
- Is the model using 2D topology, 3D coordinates, conformers, or all of them?
- Are protonation, tautomer, and charge assumptions explicit?
- Is the ligand evaluated alone or inside a [[entities/protein-ligand-complex|protein-ligand complex]]?
- Are close analogs separated from test examples when claiming ligand-side generalization?
- Is the pose generated, experimentally observed, transferred, or used only as a reference?
- Are pose quality, binding affinity, and virtual-screening rank kept as separate targets?
- Was the ligand standardized before deduplication, scaffold assignment, and featurization?
- Is the conformer or pose available at inference time?
- Are measured inactives, decoys, and unmeasured compounds kept distinct?

## Related

- [[entities/entity-relation-map|Entity relation map]]
- [[entities/molecule|Molecule]]
- [[entities/pocket|Pocket]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/molecular-modeling/protonation-state|Protonation state]]
- [[concepts/molecular-modeling/tautomer|Tautomer]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[papers/sbdd/posebusters|PoseBusters]]
