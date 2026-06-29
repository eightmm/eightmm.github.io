---
title: Molecular and Ligand Modeling
aliases:
  - computational-biology/molecular-ligand
  - computational-biology/molecules
  - bio/molecular-ligand
tags:
  - computational-biology
  - molecular-modeling
---

# Molecular and Ligand Modeling

Molecular and ligand modeling은 small-molecule identity, representation, standardization, conformer, property prediction, retrieval, generation을 다룹니다. Ligand는 molecule과 다른 object type이 아니라 binding 또는 target context 안에 있는 molecule입니다.

$$
r_L
=
\phi(L, c)
$$

여기서 $L$은 standardized molecular object이고, $c$는 pH, protonation, tautomer policy, conformer source, assay context, target, pocket을 포함할 수 있습니다.

Molecule note에서는 chemical identity를 먼저 고정하고, ligand note에서는 그 molecule이 어떤 target, pocket, assay, pose context 안에 들어갔는지 고정합니다.

## Route Map

| Question | Start | Watch |
| --- | --- | --- |
| 어떤 chemical object를 모델링하는가? | [Molecules](/molecular-modeling/molecules), [Molecule](/entities/molecule), [Ligand](/entities/ligand) | salt, mixture, stereochemistry, tautomer, protonation |
| model은 어떤 representation을 보는가? | [RDKit](/concepts/molecular-modeling/rdkit), [Molecular graph](/concepts/molecular-modeling/molecular-graph), [SMILES](/concepts/molecular-modeling/smiles), [Molecular fingerprint](/concepts/molecular-modeling/molecular-fingerprint) | equivalent molecule이 다른 input이 될 수 있음 |
| 3D conformation이 포함되는가? | [Conformer](/concepts/molecular-modeling/conformer), [Force field](/concepts/molecular-modeling/force-field) | conformer source와 minimization protocol이 지배할 수 있음 |
| molecule이 target-conditioned인가? | [Interaction Modeling](/molecular-modeling/interactions), [Target-assay-label contract](/entities/target-assay-label) | molecule-only claim은 target-specific activity를 포함하지 않음 |
| structure 또는 docking이 중심인가? | [Structure-Based Modeling](/molecular-modeling/structure-based) | pose, pocket, scoring, geometry는 별도 check가 필요함 |

## Main Subroutes

| Area | Use for | Start |
| --- | --- | --- |
| Chemical identity | standardization, salt stripping, stereo, tautomer, protonation | [Molecules](/molecular-modeling/molecules) |
| 2D representation | SMILES, molecular graph, fingerprint, scaffold, similarity | [RDKit](/concepts/molecular-modeling/rdkit), [Molecular graph](/concepts/molecular-modeling/molecular-graph) |
| 3D representation | conformer, force field, shape, geometry, docking input | [Conformer](/concepts/molecular-modeling/conformer) |
| Property and retrieval | property prediction, similarity search, candidate ranking | [Molecular property prediction](/concepts/molecular-modeling/molecular-property-prediction) |
| Generation | valid molecule samples, constrained generation, scaffold editing | [Molecular generation](/concepts/generative-models/molecular-generation) |

## Molecule vs Ligand

| Object | Meaning | Typical question |
| --- | --- | --- |
| Molecule | chemical identity after standardization | property, similarity, scaffold, generation |
| Ligand | molecule in binding/target context | pose, activity, interaction, docking input |
| Assay compound record | measured sample in an assay source | endpoint, unit, censoring, replicate |
| Conformer | 3D state of a molecule | shape, strain, docking preparation |
| Pose | ligand conformer placed in protein/pocket frame | RMSD, clash, interaction, score |

## Chemical State Checklist

| State | Ask |
| --- | --- |
| Salt / mixture | modeled molecule이 parent, salt, mixture, assay record 중 무엇인가? |
| Stereochemistry | unspecified/specified stereocenter를 일관되게 처리했는가? |
| Tautomer | canonical tautomer 하나를 택했는가, 여러 state가 가능한가? |
| Protonation / charge | state가 pH, assay, docking, force field assumption과 compatible한가? |
| Conformer | 3D geometry가 generated, experimental, minimized, complex reuse 중 무엇인가? |
| Deduplication | equivalent molecule을 split construction 전에 collapse했는가? |

## RDKit Boundary

RDKit은 molecule parsing, standardization, fingerprint, descriptor, substructure search, conformer의 implementation layer인 경우가 많습니다. 이 setting은 invisible preprocessing이 아니라 method choice로 다룹니다.

| RDKit use | Record |
| --- | --- |
| canonical SMILES | standardization and stereo policy |
| Morgan fingerprint | radius, bit length, count/binary mode, chirality flag |
| descriptor vector | descriptor list and missing-value policy |
| scaffold split | standardized molecule used for scaffold extraction |
| conformer generation | seed, conformer count, force field, minimization and failure policy |

## Claim Boundary

| Claim | Needs |
| --- | --- |
| molecule property prediction | standardized molecule, split, endpoint, metric |
| ligand activity prediction | target/assay context and label contract |
| molecular generation | validity, uniqueness, novelty, filter denominator |
| conformer quality | conformer source, energy/minimization, geometry checks |
| docking-ready ligand | protonation, tautomer, charge, atom typing, conformer policy |

## Boundary

Molecule-only 또는 ligand-preparation question은 이 페이지를 사용합니다. Row가 molecule-target-assay relation이면 [[molecular-modeling/interactions|Interaction modeling]]을 사용합니다. Pocket, pose, coordinate frame, protein-ligand complex가 claim의 일부이면 [[molecular-modeling/structure-based/index|Structure-based modeling]]을 사용합니다.

## Checks

- molecular standardization이 deduplication과 splitting 전에 수행되었는가?
- stereochemistry, charge, protonation, tautomer, conformer policy가 explicit한가?
- split이 scaffold-based, temporal, target-aware, assay-aware 중 무엇인가?
- metric이 property prediction, retrieval, generation, docking, screening 중 무엇을 test하는가?

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/entities|Objects and entities]]
- [[molecular-modeling/molecules|Molecules]]
- [[molecular-modeling/interactions|Interaction modeling]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
