---
title: Structure-Based Drug Discovery
tags:
  - sbdd
  - structure-based-modeling
---

# Structure-Based Drug Discovery

Structure-based drug discovery uses 3D molecular structure to reason about target-ligand recognition, pose quality, binding affinity, and candidate prioritization.

In this wiki, SBDD concepts support [[molecular-modeling/structure-based/index|Structure-based modeling]]. Research pages can later link back when a concrete project or thesis direction needs them.

## Route Map

| Question | Start | Then Check |
| --- | --- | --- |
| how are receptor and ligand prepared? | [Receptor and ligand preparation](/concepts/sbdd/receptor-ligand-preparation), [Docking workflow](/concepts/sbdd/docking-workflow) | chemical state, missing atoms, receptor state |
| how are poses generated? | [Pose generation](/concepts/sbdd/pose-generation), [Pose RMSD](/concepts/sbdd/pose-rmsd) | atom mapping and symmetry |
| is a predicted pose plausible? | [Pose quality](/concepts/sbdd/pose-quality), [Interaction fingerprint](/concepts/sbdd/interaction-fingerprint) | geometry, contacts, denominator |
| what score is being optimized or ranked? | [Scoring function](/concepts/sbdd/scoring-function), [Binding affinity](/concepts/sbdd/binding-affinity) | label semantics and assay context |
| is this a screening claim? | [Virtual screening](/concepts/sbdd/virtual-screening), [Negative set](/concepts/evaluation/negative-set), [Ranking metrics](/concepts/evaluation/ranking-metrics) | candidate pool and early enrichment |
| is the benchmark clean? | [Protein-ligand split](/concepts/sbdd/protein-ligand-split), [Template leakage](/concepts/sbdd/template-leakage) | protein family, scaffold, complex, assay/source |

## Core Concepts

- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/pose-rmsd|Pose RMSD]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/sbdd/docking-workflow|Docking workflow]]
- [[concepts/sbdd/receptor-ligand-preparation|Receptor and ligand preparation]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/interaction-fingerprint|Interaction fingerprint]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[concepts/sbdd/template-leakage|Template leakage]]
- [[entities/pocket|Pocket]]

## Task Map

$$
(P, L)
\rightarrow
\{\text{candidate poses}\}
\rightarrow
\{\text{pose validity}, \text{score}, \text{affinity}, \text{rank}\}
$$

where $P$ is a protein or pocket and $L$ is a ligand.

## Checks

- Is the task pose prediction, affinity prediction, enrichment, or molecule generation?
- Is pose generation evaluated separately from scoring?
- Are receptor and ligand inputs prepared consistently?
- Are pose quality and binding affinity evaluated separately?
- Is pose RMSD symmetry-corrected and separated from interaction or affinity claims?
- Does the benchmark split test scaffold, protein-family, temporal, or structure-level generalization?
- Is the protein-ligand split policy aligned with the claim?
- Could training data or template databases leak a close protein, ligand, or bound complex into evaluation?
- Are invalid generated structures filtered before ranking claims?

## Related

- [[molecular-modeling/structure-based/index|Structure-based modeling]]
- [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/pose-rmsd|Pose RMSD]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[papers/sbdd/posebusters|PoseBusters]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/protein-modeling/binding-site|Binding site]]
