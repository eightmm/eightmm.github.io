---
title: Molecular Modeling Concepts
tags:
  - molecular-modeling
  - concepts
---

# Molecular Modeling Concepts

Molecular modeling concept는 small molecule이 string, graph, fingerprint, conformer, descriptor, 3D coordinate, physics-based geometry protocol 같은 model input으로 바뀌는 과정을 설명합니다.

Modeling object는 단순한 atom 그림이 아니라 stateful record입니다.

$$
M =
(\text{topology}, \text{stereo}, \text{tautomer}, \text{protonation}, \text{conformer}, \text{source})
$$

선택이 달라지면 deduplication key, split assignment, feature, pose, label이 달라질 수 있습니다.

## Workflow

공개 ML note에서는 아래 순서를 우선합니다.

$$
\text{raw record}
\rightarrow
\text{standardize}
\rightarrow
\text{define identity}
\rightarrow
\text{deduplicate}
\rightarrow
\text{split}
\rightarrow
\text{featurize}
\rightarrow
\text{model}
$$

Molecular identity policy가 명확해지기 전에는 label을 split하거나 aggregate하지 않습니다.

## 이동 지도

| 질문 | 시작점 | 위험 |
| --- | --- | --- |
| What is the molecule identity? | [Molecular identity](/concepts/molecular-modeling/molecular-identity), [Molecular standardization](/concepts/molecular-modeling/molecular-standardization), [Chemical state contract](/concepts/molecular-modeling/chemical-state-contract) | salts, tautomers, stereo, protonation, charge, source policy |
| What does the model see? | [Molecular featurization contract](/concepts/molecular-modeling/molecular-featurization-contract), [RDKit](/concepts/molecular-modeling/rdkit), [SMILES](/concepts/molecular-modeling/smiles), [Molecular graph](/concepts/molecular-modeling/molecular-graph), [Molecular fingerprint](/concepts/molecular-modeling/molecular-fingerprint) | featurizer silently changing the object |
| How is similarity or retrieval defined? | [Molecular similarity](/concepts/molecular-modeling/molecular-similarity), [Substructure search](/concepts/molecular-modeling/substructure-search) | proxy similarity used as domain truth |
| What prediction task is claimed? | [Molecular property prediction](/concepts/molecular-modeling/molecular-property-prediction) | label context and split unit missing |
| Is generation constrained? | [Fragment-SELFIES](/concepts/molecular-modeling/fragment-selfies) | valid strings without useful molecules |
| Is 3D state involved? | [Conformer](/concepts/molecular-modeling/conformer), [Force field](/concepts/molecular-modeling/force-field), [Energy minimization](/concepts/molecular-modeling/energy-minimization), [Molecular dynamics](/concepts/molecular-modeling/molecular-dynamics) | conformer source and postprocessing dependence |
| Which chemical variants matter? | [Tautomer](/concepts/molecular-modeling/tautomer), [Protonation state](/concepts/molecular-modeling/protonation-state), [Stereochemistry](/concepts/molecular-modeling/stereochemistry) | train/test leakage through equivalent or near-equivalent raw rows |

## Geometry와 Physics Protocol

| 개념 | 용도 | 주요 위험 |
| --- | --- | --- |
| [Conformer](/concepts/molecular-modeling/conformer) | ligand 3D geometry and conformer ensembles | training and inference may use different conformer sources |
| [Force field](/concepts/molecular-modeling/force-field) | geometry energy, minimization, MD, clash checks | energy is model-dependent, not an absolute truth |
| [Energy minimization](/concepts/molecular-modeling/energy-minimization) | relaxing conformers or poses | postprocessing can hide invalid model outputs |
| [Molecular dynamics](/concepts/molecular-modeling/molecular-dynamics) | trajectories and flexible structure analysis | frame leakage and protocol dependence can dominate |

## Data check

- deduplication과 split 전에 molecule을 standardize합니다.
- deduplication, label aggregation, split assignment 전에 molecular identity를 정의합니다.
- stereochemistry를 보존할지 flatten할지 정합니다.
- tautomer, salt, charge, protonation, conformer protocol을 기록합니다.
- coordinate를 생성하거나 refine하면 force-field, minimization, molecular-dynamics protocol을 기록합니다.
- ligand-side generalization에는 random split보다 scaffold 또는 cluster split을 씁니다.
- feature는 featurizer version과 input hash로 cache합니다.
- train, evaluation, inference에서 하나의 molecular featurization contract를 씁니다.
- RDKit parsing, sanitization, fingerprinting, descriptor, conformer setting은 method의 일부로 취급합니다.

## Failure mode

- salt, tautomer, charge, stereo variant가 서로 다른 raw row로 train/test에 새어 들어갑니다.
- label은 stereoisomer를 구분하지만 graph featurizer가 chiral tag나 bond stereo를 버립니다.
- 3D model이 crystal/bound conformation에서 학습하고 generated conformer에서 deploy되지만 shift를 측정하지 않습니다.
- energy minimization postprocessing이 model output을 바꾸는데 raw model 결과처럼 보고됩니다.
- similarity 또는 scaffold split이 standardized identity가 아니라 raw molecule에서 계산됩니다.
- congeneric series에 random split을 써서 memorization이 generalization처럼 보입니다.

## Related

- [[entities/molecule|Molecule]]
- [[entities/ligand|Ligand]]
- [[entities/target-assay-label|Target-assay-label]]
- [[concepts/molecular-modeling/rdkit|RDKit]]
- [[concepts/sbdd/index|Structure-based drug discovery]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/activity-cliff|Activity cliff]]
