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

Interaction modeling은 molecule 하나나 protein 하나가 아니라 biological object와 chemical object 사이의 relation을 예측 단위로 삼는 문제를 다룹니다. 대표 예시는 target-conditioned activity, binding affinity, selectivity, protein-ligand interaction, protein-protein interaction, complex-level prediction입니다.

핵심 object는 아래 tuple입니다.

$$
u
=
(L,\ T,\ A,\ C)
$$

여기서 $L$은 ligand 또는 molecule, $T$는 target 또는 protein, $A$는 assay 또는 measurement context, $C$는 optional structure, pocket, species, construct, mutation, source context입니다.

The prediction is:

$$
\hat{y}
=
f_\theta(r_L,\ r_T,\ r_A,\ r_C)
$$

각 representation이 정의되어야 label이나 metric을 믿을 수 있습니다.

## Route Map

| Question | Start | Watch |
| --- | --- | --- |
| 어떤 object들이 interact하는가? | [Entities](/molecular-modeling/entities), [Protein](/entities/protein), [Ligand](/entities/ligand), [Target](/entities/target) | target identity가 isoform, construct, mutation, species를 숨길 수 있음 |
| label이 assay-conditioned인가? | [Target-assay-label contract](/entities/target-assay-label), [Bioactivity label](/entities/bioactivity-label), [Assay](/entities/assay) | endpoint, unit, censoring, threshold, source mismatch |
| interaction이 structural한가? | [Protein-ligand complex](/entities/protein-ligand-complex), [Pocket](/entities/pocket), [Protein-ligand interaction](/concepts/sbdd/protein-ligand-interaction) | ligand-defined pocket 또는 known pose가 inference에서 unavailable할 수 있음 |
| output이 scalar relation인가? | [Interaction prediction](/concepts/tasks/interaction-prediction), [Binding affinity](/concepts/sbdd/binding-affinity) | score, probability, affinity, ranking은 서로 다른 claim |
| split은 어떻게 구성해야 하는가? | [Protein-ligand split](/concepts/sbdd/protein-ligand-split), [Scaffold split](/concepts/evaluation/scaffold-split), [Protein family split](/concepts/evaluation/protein-family-split) | pair만 hold out해도 ligand/protein neighborhood가 leak될 수 있음 |

## Interaction Types

| Type | Unit | Typical output | Main risk |
| --- | --- | --- | --- |
| Target-conditioned activity | molecule-target-assay record | active/inactive, pIC50, Ki, Kd, EC50 | assay/source conflict and label direction errors |
| Binding affinity | protein-ligand pair or complex | $\Delta G$, $K_d$, $K_i$, rank score | units, temperature, and experimental protocol mismatch |
| Pose-aware interaction | protein, pocket, ligand, pose | pose quality, contact pattern, interaction fingerprint | known-pose or ligand-frame leakage |
| Selectivity | molecule across several targets | target-specific ranking or margin | target panel and missing-label bias |
| Protein-protein interaction | protein pair or complex | probability, interface, contact map | homolog and complex-family leakage |

## Representation Contract

각 side에서 어떤 representation을 쓰는지 명시합니다.

| Side | Examples | Check |
| --- | --- | --- |
| Ligand | SMILES, molecular graph, fingerprint, conformer, docked pose | deduplication과 split 전에 standardized되어야 함 |
| Protein | sequence, protein embedding, structure, pocket graph, surface | sequence identity와 structure source를 기록해야 함 |
| Assay | endpoint, organism, construct, units, threshold, source | generic label 하나로 뭉개면 안 됨 |
| Context | pocket, template, species, mutation, conformer source | inference time에 사용 가능한 정보여야 함 |

Pair model에서 feature map은 보통 아래처럼 씁니다.

$$
r_{LT}
=
\phi(r_L,\ r_T,\ r_C)
$$

Fusion method는 concatenation, cross-attention, graph construction, interaction fingerprint, structure-aware complex graph 등이 될 수 있습니다.

## Score Semantics

Interaction score는 하나의 숫자처럼 보여도 의미가 다릅니다.

| Score | 의미 | 혼동하지 말 것 |
| --- | --- | --- |
| activity probability | assay threshold 아래의 active/inactive decision | binding affinity |
| affinity value | $K_d$, $K_i$, IC50, $\Delta G$ 형태의 measurement | pose quality |
| docking score | ranking 또는 heuristic energy proxy | calibrated probability |
| enrichment score | early retrieval success | absolute affinity |
| selectivity margin | target panel difference | single-target activity |

## Split and Leakage

Interaction claim에는 최소 하나의 explicit holdout axis가 필요합니다.

| Claim | 필요한 split |
| --- | --- |
| known target에 대한 new molecule | scaffold 또는 chemical-series split |
| new protein에 대한 known molecule | protein-family 또는 sequence-identity split |
| new molecule과 new protein | scaffold split과 protein-family split |
| new assay/source | assay, source, campaign split |
| new structure template | template-aware 및 homolog-aware structure split |

Random row split에서 broad interaction generalization을 주장하면 안 됩니다. Row split은 train/test 사이에 같은 ligand series, homologous protein, related assay, 거의 같은 complex template을 남길 수 있습니다.

## Metrics

| Task | Primary metric | Diagnostic |
| --- | --- | --- |
| Binary activity | PRC-AUC, enrichment, calibrated threshold metric | ROC-AUC, reliability, class prevalence |
| Affinity regression | MAE/RMSE in original units, Spearman/Pearson | assay slice error, activity-cliff error |
| Ranking or screening | enrichment, BEDROC, recall at budget | decoy bias, negative provenance |
| Pose-aware interaction | pose validity, contact recovery, interaction fingerprint similarity | clash, strain, ligand-state mismatch |

## Checks

- one example은 molecule, target, assay row, pair, complex, pose, ranked list 중 무엇인가?
- target, assay, endpoint, unit, threshold, censoring, source가 보존되는가?
- ligand scaffold, protein family, assay/source, template leakage를 따로 확인했는가?
- metric이 activity prediction, affinity regression, screening, pose evaluation, selectivity 중 실제 decision과 맞는가?
- model이 inference time에 unavailable한 pocket, pose, template, assay field를 쓰는가?
- fingerprint model, sequence-similarity baseline, docking baseline, nearest-neighbor baseline 같은 cheap baseline이 있는가?

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/proteins|Proteins]]
- [[molecular-modeling/molecules|Molecules]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
- [[molecular-modeling/data-evaluation|Data and evaluation]]
- [[concepts/tasks/interaction-prediction|Interaction prediction]]
