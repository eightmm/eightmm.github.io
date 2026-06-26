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

Expanded as a contract:

$$
\begin{aligned}
(P_{\mathrm{raw}},L_{\mathrm{raw}})
&\xrightarrow{\phi_{\mathrm{prep}}}
(P,L) \\
(P,L)
&\xrightarrow{\phi_{\mathrm{search}}}
\{\hat{X}_k\}_{k=1}^{K} \\
(P,L,\hat{X}_k)
&\xrightarrow{f_{\mathrm{score}}}
s_k \\
\{(X_k,s_k)\}_{k=1}^{K}
&\xrightarrow{\phi_{\mathrm{select}}}
\hat{X}_{\mathrm{top}}
\end{aligned}
$$

Each arrow is a separate source of assumptions and failure.

## Key Ideas

- Pose generation and scoring are different tasks.
- A good score is not enough if the generated pose is physically implausible.
- Docking boxes, protonation, conformer generation, and protein preparation can dominate results.
- Force-field minimization or pose relaxation should be reported as part of the method if it changes the output.
- Receptor and ligand preparation should be treated as part of the method, not as invisible preprocessing.
- Learned scoring functions should be evaluated separately from search and filtering heuristics.
- Public notes should describe generic workflow decisions, not private targets or unpublished results.

## Workflow Variants

| Variant | Main output | Typical evidence |
|---|---|---|
| pose prediction | top pose geometry | pose RMSD, PoseBusters-style validity |
| virtual screening | ranked compounds | enrichment, AUPRC, recall at budget |
| affinity prediction | binding score or $\Delta G$ | regression/correlation and split control |
| rescoring | new ranking over fixed poses | pose source and score-only ablation |
| docking plus relaxation | refined pose | report raw and relaxed metrics separately |

Do not compare workflows unless preparation, search budget, pose count, and failure handling are aligned.

## Claim Separation

| Claim | Required separation |
|---|---|
| "better search" | same scoring and filtering, different pose generation |
| "better scoring" | same generated poses, different score function |
| "better preparation" | same search/scoring, different preparation policy |
| "better end-to-end docking" | report each component and the final metric |

This prevents a score function from receiving credit for better search settings, or a pose generator from receiving credit for a hidden filter.

## Practical Checks

- What protein structure and binding site definition are used?
- How are ligand states, conformers, charges, and stereochemistry prepared?
- How many poses are generated and how is diversity handled?
- Are poses minimized or relaxed after generation, and are raw invalid poses counted?
- Is the workflow evaluating pose prediction, enrichment, affinity, or prioritization?
- Are invalid poses filtered before ranking metrics are reported?
- Are baselines and split protocols appropriate for the generalization claim?
- Is the comparison budget-matched: number of poses, search time, conformers, and receptor states?
- Are pose prediction, affinity prediction, and screening metrics reported separately?
- Are preparation failures included in the denominator?

## Related

- [[molecular-modeling/structure-based/protein-ligand-docking|Protein-ligand docking]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/pose-rmsd|Pose RMSD]]
- [[concepts/sbdd/receptor-ligand-preparation|Receptor and ligand preparation]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/protein-modeling/binding-site|Binding site]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[concepts/molecular-modeling/force-field|Force field]]
- [[concepts/molecular-modeling/energy-minimization|Energy minimization]]
- [[papers/sbdd/posebusters|PoseBusters]]
