---
title: Docking
aliases:
  - computational-biology/docking
  - bio/docking
tags:
  - computational-biology
  - docking
  - structure-based-modeling
unlisted: true
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

A pose can be parameterized as translation, rotation, and internal torsions:

$$
q = (t, R, \tau),
\qquad
X(q) = R X_L(\tau) + \mathbf{1}t^\top
$$

where $t\in\mathbb{R}^3$ places the ligand, $R\in SO(3)$ orients it, and $\tau$ controls rotatable bonds. Docking searches over $q$ under a receptor or pocket context.

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

## Object Contract

Docking examples should state the object boundary before model claims.

| Field | Meaning |
| --- | --- |
| Protein or pocket | full receptor, cleaned receptor, predicted pocket, known binding site, or blind search region |
| Ligand identity | standardized molecule, tautomer/protonation/stereo state, conformer source |
| Pose target | crystallographic pose, curated reference pose, generated candidate pose, or no pose label |
| Search space | fixed pocket, flexible side chains, ligand flexibility, constraints, or blind docking |
| Scoring target | pose ranking, candidate ranking, affinity proxy, enrichment, or downstream selection |
| Split unit | ligand scaffold, protein family, complex pair, assay/source, or time |

## Separation of Claims

Do not collapse these claims:

- pose generation: can the method place the ligand plausibly?
- pose quality: is the geometry chemically and physically plausible?
- scoring: does the method rank poses or candidates well?
- affinity prediction: does the method estimate binding-related labels?
- virtual screening: does the method enrich useful candidates early?

The same method can be strong for one claim and weak for another. A low pose RMSD does not automatically prove affinity prediction, and a high enrichment score does not automatically prove chemically valid poses.

## Search and Scoring

Docking often combines a search distribution and a score:

$$
q_k \sim Q(q\mid P,L,c),
\qquad
s_k = S(P,L,X(q_k))
$$

The selected pose is usually:

$$
\hat{q}
=
\operatorname*{arg\,min}_{q_k}
S(P,L,X(q_k))
$$

or $\operatorname*{arg\,max}$ if the score convention is higher-is-better. This convention must be written explicitly.

## Typical Metrics

| Claim | Typical Metric | Caveat |
| --- | --- | --- |
| Pose generation | RMSD to reference pose | symmetry correction and atom mapping matter |
| Pose plausibility | geometry checks, steric clashes, bond validity, interaction sanity | a plausible pose can still rank poorly |
| Scoring / ranking | Spearman, Kendall, top-k success, enrichment | score calibration is separate from ranking |
| Affinity prediction | RMSE, MAE, Pearson/Spearman, calibration diagnostics | assay noise and label semantics dominate |
| Virtual screening | enrichment factor, BEDROC, ROC-AUC, PR-AUC | active/decoy construction can bias conclusions |

## What Goes to Papers

Keep this page for reusable docking concepts. Put a page under [[papers/sbdd/index|Structure-based modeling papers]] when the note is about a specific docking method, benchmark, or empirical result. For example, PoseBusters belongs in Papers because it is a specific evaluation paper; pose plausibility checks belong here and in [[concepts/sbdd/pose-quality|Pose quality]].

## Checks

- Is the pocket known, predicted, ligand-defined, or blind?
- Are ligand scaffold and protein family split units both considered?
- Are protonation, tautomer, stereochemistry, and conformer choices recorded?
- Is RMSD symmetry-corrected and separated from interaction quality?
- Is a docking score being treated as affinity without evidence?

## Related

- [[molecular-modeling/structure-based/index|Structure-based modeling]]
- [[concepts/sbdd/docking-workflow|Docking workflow]]
- [[concepts/sbdd/index|SBDD concepts]]
- [[molecular-modeling/data-evaluation|Data and evaluation]]
