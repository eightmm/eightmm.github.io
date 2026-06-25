---
title: Docking Workflow
tags:
  - sbdd
  - docking
  - workflow
---

# Docking Workflow

A docking workflow estimates plausible ligand poses in a protein binding site and scores or filters them for downstream use. A useful public note should separate pose generation, pose quality checks, scoring, and ranking claims.

A simplified workflow is:

$$
(P,L)
\rightarrow
\{\hat{X}_1,\ldots,\hat{X}_K\}
\rightarrow
\{s_1,\ldots,s_K\}
\rightarrow
\operatorname{rank}
$$

where $P$ is a protein or pocket, $L$ is a ligand, $\hat{X}_k$ are candidate poses, and $s_k$ are scores.

The first arrow is [[concepts/sbdd/pose-generation|pose generation]]. The second arrow is scoring or filtering. Keeping those steps separate makes failures diagnosable.

## Key Ideas

- Pose generation and scoring are different tasks.
- A good score is not enough if the generated pose is physically implausible.
- Docking boxes, protonation, conformer generation, and protein preparation can dominate results.
- Receptor and ligand preparation should be treated as part of the method, not as invisible preprocessing.
- Learned scoring functions should be evaluated separately from search and filtering heuristics.
- Public notes should describe generic workflow decisions, not private targets or unpublished results.

## Practical Checks

- What protein structure and binding site definition are used?
- How are ligand states, conformers, charges, and stereochemistry prepared?
- How many poses are generated and how is diversity handled?
- Is the workflow evaluating pose prediction, enrichment, affinity, or prioritization?
- Are invalid poses filtered before ranking metrics are reported?
- Are baselines and split protocols appropriate for the generalization claim?

## Related

- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/receptor-ligand-preparation|Receptor and ligand preparation]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/protein-modeling/binding-site|Binding site]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[papers/sbdd/posebusters|PoseBusters]]
