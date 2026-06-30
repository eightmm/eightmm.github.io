---
title: Structure-Based Drug Discovery
tags:
  - sbdd
  - structure-based-modeling
---

# Structure-Based Drug Discovery

Structure-based drug discovery는 3D molecular structure를 사용해 target-ligand recognition, pose quality, binding affinity, candidate prioritization을 해석합니다.

이 wiki에서 SBDD concept는 [[molecular-modeling/structure-based/index|Structure-based modeling]]을 받쳐주는 층입니다. 구체적인 project나 thesis direction이 생기면 Research page가 여기로 다시 연결됩니다.

## SBDD Contract

SBDD note는 protein-ligand system을 아래 계약으로 고정한 뒤 method와 claim을 읽습니다.

$$
\mathcal{S}_{\mathrm{SBDD}}
=
(P,\ B,\ L,\ X,\ Q,\ s,\ m,\ \Delta)
$$

| Part | Meaning | Typical question |
| --- | --- | --- |
| $P$ | protein or receptor | structure source, chain, missing residues, cofactors, protonation? |
| $B$ | binding site or pocket | known, predicted, ligand-defined, blind, template-derived? |
| $L$ | ligand or candidate set | stereo, tautomer, charge, conformer policy, library source? |
| $X$ | pose or complex geometry | reference, docked, generated, minimized, noisy, refined? |
| $Q$ | pose quality rule | RMSD, clash, strain, interaction, geometry validity? |
| $s$ | score or predictor | pose score, affinity, ranking score, filter, uncertainty? |
| $m$ | metric | pose success, enrichment, correlation, calibration, validity, utility? |
| $\Delta$ | split and leakage boundary | scaffold, protein family, complex pair, template, time, assay source? |

The first reading question is not whether a method is AI-based. It is which part of the contract it changes.

$$
\text{method}
\in
\{\text{preparation}, \text{pocket}, \text{pose generation}, \text{scoring}, \text{ranking}, \text{evaluation}\}
$$

## Route Map

| Question | Start | Then Check |
| --- | --- | --- |
| how are receptor and ligand prepared? | [Receptor and ligand preparation](/concepts/sbdd/receptor-ligand-preparation), [Docking workflow](/concepts/sbdd/docking-workflow) | chemical state, missing atoms, receptor state |
| how is the pocket defined? | [Pocket definition contract](/concepts/sbdd/pocket-definition-contract) | known, predicted, ligand-defined, blind, template-derived |
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
- [[concepts/sbdd/pocket-definition-contract|Pocket definition contract]]
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

## Claim Separation

SBDD papers often use similar inputs but make different claims. Keep the claim narrow.

| Claim | Evidence needed | Do not confuse with |
| --- | --- | --- |
| pose generation | reference pose, symmetry-aware RMSD, pose plausibility | affinity or enrichment |
| pose plausibility | clash, strain, interaction, chemical geometry | correct binding free energy |
| scoring | score separates better/worse poses or ligands | calibrated affinity |
| affinity prediction | assay-aware labels, split, uncertainty, baseline | docking score |
| virtual screening | candidate pool, negative set, early enrichment | absolute activity |
| structure-aware generation | generated validity, novelty, conditioning, filters | synthesizability or selectivity |

Denominators matter:

$$
\text{useful hit rate}
=
\frac{\#\text{valid and useful candidates}}
{\#\text{attempted candidates before filtering}}
$$

Filtering invalid poses or molecules before reporting ranking metrics changes the claim and must be stated.

## Evaluation Denominators

| Evaluation | Numerator | Denominator |
| --- | --- | --- |
| pose success | poses within threshold and passing plausibility checks | all generated or docked poses attempted |
| docking enrichment | actives recovered in early ranked set | full screened library or declared candidate pool |
| affinity correlation | examples with valid label and prediction | declared benchmark after preprocessing |
| generation validity | samples passing chemistry and geometry rules | raw generated samples before repair/filtering |
| interaction recovery | contacts or fingerprints matching reference | evaluated complexes with defined pocket and pose |

## Public Note Template

| Field | Write |
| --- | --- |
| System | protein, pocket, ligand, complex, structure source |
| Preparation | receptor cleanup, ligand state, conformer policy, pocket rule |
| Method | docking, scoring, diffusion, refinement, screening, generation |
| Output | pose, score, rank, affinity, candidate set, generated molecule |
| Metric | RMSD, clash, strain, enrichment, correlation, calibration, validity |
| Split | scaffold, protein family, complex pair, template, assay/source, time |
| Leakage audit | ligand-defined pocket, template similarity, benchmark overlap |
| Public boundary | no private structures, internal target names, unpublished results, server paths |

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
- [[concepts/sbdd/pocket-definition-contract|Pocket definition contract]]
- [[concepts/sbdd/pose-rmsd|Pose RMSD]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[papers/sbdd/posebusters|PoseBusters]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/protein-modeling/binding-site|Binding site]]
