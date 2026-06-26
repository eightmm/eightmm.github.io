---
title: Docking
aliases:
  - bio-ai/docking
tags:
  - bio
  - docking
  - structure-based-ai
---

# Docking

Docking estimates plausible ligand poses in a protein binding site and often ranks candidates for structure-based screening. In this site, docking is treated as a workflow with separate preparation, search, scoring, filtering, and evaluation steps.

$$
(P, L, c)
\rightarrow
\{\hat{X}_1,\ldots,\hat{X}_K\}
\rightarrow
\operatorname{rank}(\hat{X}_k)
$$

where $P$ is the protein or pocket, $L$ is the ligand, $c$ is context such as pocket definition or constraints, and $\hat{X}_k$ are candidate ligand poses.

## Workflow Map

| Step | Read |
| --- | --- |
| Define the object | [Protein-ligand complex](/entities/protein-ligand-complex), [Pocket](/entities/pocket), [Ligand](/entities/ligand) |
| Prepare inputs | [Receptor and ligand preparation](/concepts/sbdd/receptor-ligand-preparation) |
| Generate poses | [Pose generation](/concepts/sbdd/pose-generation) |
| Score candidates | [Scoring function](/concepts/sbdd/scoring-function), [Binding affinity](/concepts/sbdd/binding-affinity) |
| Check geometry | [Pose quality](/concepts/sbdd/pose-quality), [Pose RMSD](/concepts/sbdd/pose-rmsd), [PoseBusters](/papers/sbdd/posebusters) |
| Screen libraries | [Virtual screening](/concepts/sbdd/virtual-screening), [Interaction fingerprint](/concepts/sbdd/interaction-fingerprint) |
| Check generalization | [Protein-ligand split](/concepts/sbdd/protein-ligand-split), [Template leakage](/concepts/sbdd/template-leakage) |

## Separation of Claims

Do not collapse these claims:

- pose generation: can the method place the ligand plausibly?
- pose quality: is the geometry chemically and physically plausible?
- scoring: does the method rank poses or candidates well?
- affinity prediction: does the method estimate binding-related labels?
- virtual screening: does the method enrich useful candidates early?

## Checks

- Is the pocket known, predicted, ligand-defined, or blind?
- Are ligand scaffold and protein family split units both considered?
- Are protonation, tautomer, stereochemistry, and conformer choices recorded?
- Is RMSD symmetry-corrected and separated from interaction quality?
- Is a docking score being treated as affinity without evidence?

## Related

- [[bio/structure-based-ai|Structure-based AI]]
- [[research/structure-based-ai/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/index|SBDD concepts]]
- [[bio/data-evaluation|Data and evaluation]]
