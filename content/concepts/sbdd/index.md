---
title: Structure-Based Drug Discovery
tags:
  - sbdd
  - structure-based-ai
---

# Structure-Based Drug Discovery

Structure-based drug discovery uses 3D molecular structure to reason about target-ligand recognition, pose quality, binding affinity, and candidate prioritization.

In this wiki, SBDD concepts are reusable notes under [[research/structure-based-ai/index|Structure-based AI]].

## Core Concepts

- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/sbdd/docking-workflow|Docking workflow]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/interaction-fingerprint|Interaction fingerprint]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[entities/pocket|Pocket]]

## Task Map

$$
(P, L)
\rightarrow
\{\text{pose}, \text{score}, \text{affinity}, \text{rank}\}
$$

where $P$ is a protein or pocket and $L$ is a ligand.

## Checks

- Is the task pose prediction, affinity prediction, enrichment, or molecule generation?
- Are pose quality and binding affinity evaluated separately?
- Does the benchmark split test scaffold, protein-family, temporal, or structure-level generalization?
- Are invalid generated structures filtered before ranking claims?

## Related

- [[research/structure-based-ai/index|Structure-based AI]]
- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
- [[papers/sbdd/posebusters|PoseBusters]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/protein-modeling/binding-site|Binding site]]
