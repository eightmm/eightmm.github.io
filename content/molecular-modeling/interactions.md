---
title: Interaction Modeling
aliases:
  - computational-biology/interactions
  - bio/interactions
tags:
  - computational-biology
  - interactions
---

# Interaction Modeling

Interaction modeling covers predictions whose unit is not a molecule alone or a protein alone, but a relation between biological and chemical objects. Typical examples are target-conditioned activity, binding affinity, selectivity, protein-ligand interaction, protein-protein interaction, and complex-level prediction.

The main object is a tuple:

$$
u
=
(L,\ T,\ A,\ C)
$$

where $L$ is a ligand or molecule, $T$ is a target or protein, $A$ is an assay or measurement context, and $C$ is optional structure, pocket, species, construct, mutation, or source context.

The prediction is:

$$
\hat{y}
=
f_\theta(r_L,\ r_T,\ r_A,\ r_C)
$$

where each representation must be defined before the label or metric is trusted.

## Route Map

| Question | Start | Watch |
| --- | --- | --- |
| What objects interact? | [Entities](/molecular-modeling/entities), [Protein](/entities/protein), [Ligand](/entities/ligand), [Target](/entities/target) | target identity may hide isoform, construct, mutation, or species |
| Is the label assay-conditioned? | [Target-assay-label contract](/entities/target-assay-label), [Bioactivity label](/entities/bioactivity-label), [Assay](/entities/assay) | endpoint, unit, censoring, threshold, and source mismatch |
| Is the interaction structural? | [Protein-ligand complex](/entities/protein-ligand-complex), [Pocket](/entities/pocket), [Protein-ligand interaction](/concepts/sbdd/protein-ligand-interaction) | ligand-defined pocket or known pose unavailable at inference |
| Is the output a scalar relation? | [Interaction prediction](/concepts/tasks/interaction-prediction), [Binding affinity](/concepts/sbdd/binding-affinity) | score, probability, affinity, and ranking are different claims |
| How should the split be constructed? | [Protein-ligand split](/concepts/sbdd/protein-ligand-split), [Scaffold split](/concepts/evaluation/scaffold-split), [Protein family split](/concepts/evaluation/protein-family-split) | holding out only the pair can still leak ligand and protein neighborhoods |

## Interaction Types

| Type | Unit | Typical Output | Main Risk |
| --- | --- | --- | --- |
| Target-conditioned activity | molecule-target-assay record | active/inactive, pIC50, Ki, Kd, EC50 | assay/source conflict and label direction errors |
| Binding affinity | protein-ligand pair or complex | $\Delta G$, $K_d$, $K_i$, rank score | units, temperature, and experimental protocol mismatch |
| Pose-aware interaction | protein, pocket, ligand, pose | pose quality, contact pattern, interaction fingerprint | known-pose or ligand-frame leakage |
| Selectivity | molecule across several targets | target-specific ranking or margin | target panel and missing-label bias |
| Protein-protein interaction | protein pair or complex | probability, interface, contact map | homolog and complex-family leakage |

## Representation Contract

State which representation is used on each side:

| Side | Examples | Check |
| --- | --- | --- |
| Ligand | SMILES, molecular graph, fingerprint, conformer, docked pose | standardized before deduplication and split |
| Protein | sequence, protein embedding, structure, pocket graph, surface | sequence identity and structure source recorded |
| Assay | endpoint, organism, construct, units, threshold, source | not collapsed into a generic label |
| Context | pocket, template, species, mutation, conformer source | available at inference time |

For pair models, the feature map is often:

$$
r_{LT}
=
\phi(r_L,\ r_T,\ r_C)
$$

The fusion method can be concatenation, cross-attention, graph construction, interaction fingerprint, or a structure-aware complex graph.

## Score Semantics

Interaction score는 하나의 숫자처럼 보여도 의미가 다릅니다.

| Score | Meaning | Do Not Confuse With |
| --- | --- | --- |
| activity probability | active/inactive decision under assay threshold | binding affinity |
| affinity value | $K_d$, $K_i$, IC50, $\Delta G$ style measurement | pose quality |
| docking score | ranking or heuristic energy proxy | calibrated probability |
| enrichment score | early retrieval success | absolute affinity |
| selectivity margin | target panel difference | single-target activity |

## Split and Leakage

Interaction claims need at least one explicit holdout axis.

| Claim | Split Needed |
| --- | --- |
| New molecule against known targets | scaffold or chemical-series split |
| Known molecule against new protein | protein-family or sequence-identity split |
| New molecule and new protein | scaffold split plus protein-family split |
| New assay/source | assay, source, or campaign split |
| New structure template | template-aware and homolog-aware structure split |

Do not claim broad interaction generalization from a random row split. A row split can preserve the same ligand series, homologous proteins, related assays, or nearly identical complex templates across train and test.

## Metrics

| Task | Primary Metric | Diagnostics |
| --- | --- | --- |
| Binary activity | PRC-AUC, enrichment, calibrated threshold metric | ROC-AUC, reliability, class prevalence |
| Affinity regression | MAE/RMSE in original units, Spearman/Pearson | assay slice error, activity-cliff error |
| Ranking or screening | enrichment, BEDROC, recall at budget | decoy bias, negative provenance |
| Pose-aware interaction | pose validity, contact recovery, interaction fingerprint similarity | clash, strain, ligand-state mismatch |

## Checks

- Is one example a molecule, target, assay row, pair, complex, pose, or ranked list?
- Are target, assay, endpoint, unit, threshold, censoring, and source preserved?
- Are ligand scaffold, protein family, assay/source, and template leakage checked separately?
- Is the metric aligned with the decision: activity prediction, affinity regression, screening, pose evaluation, or selectivity?
- Does the model use a pocket, pose, template, or assay field that would be unavailable at inference time?
- Is there a cheap baseline such as fingerprint model, sequence-similarity baseline, docking baseline, or nearest-neighbor baseline?

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/proteins|Proteins]]
- [[molecular-modeling/molecules|Molecules]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
- [[molecular-modeling/data-evaluation|Data and evaluation]]
- [[concepts/tasks/interaction-prediction|Interaction prediction]]
