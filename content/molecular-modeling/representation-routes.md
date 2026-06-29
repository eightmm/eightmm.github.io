---
title: Representation Routes
tags:
  - computational-biology
  - representation
---

# Representation Routes

Computational Biology에서는 "무슨 모델을 쓸까"보다 "무엇을 어떤 representation으로 바꿀까"가 먼저입니다. 같은 protein-ligand 문제라도 sequence, graph, conformer, pocket, complex graph, coordinate representation은 서로 다른 claim을 만듭니다.

$$
r:
\text{biological or chemical object}
\rightarrow
\text{model input}
$$

Representation route는 model choice보다 앞섭니다. 모델이 아무리 강해도 inference time에 없는 정보, split을 새게 만드는 정보, label을 암시하는 preprocessing을 입력에 넣으면 claim이 무너집니다.

## Route Map

| Route | Input unit | Good for | Watch |
| --- | --- | --- | --- |
| Sequence | amino acid, nucleotide, SMILES token | language modeling, motif, protein family, sequence-only prediction | homolog, duplicate, tokenization leakage |
| Fingerprint | molecule identity or substructure bits | fast property baseline, retrieval, similarity | stereochemistry, tautomer, protonation loss |
| Molecular graph | atom and bond graph | property, interaction, generation | standardization, aromaticity, charge policy |
| Conformer | 3D ligand geometry | docking, shape, strain, conformer-sensitive property | conformer generator and energy policy |
| Pocket | binding-site residues or grid | target-conditioned scoring, docking, retrieval | ligand-defined pocket leakage |
| Complex graph | protein-ligand pair with edges | interaction, pose, affinity, message passing | pair split and edge construction leakage |
| Coordinates | atom/residue positions | pose, structure, force, geometry-aware generation | frame, units, alignment, equivariance |

## Route to Architecture

| Representation | Typical architecture | Key inductive bias |
| --- | --- | --- |
| Sequence tokens | Transformer, state-space model, RNN | order, context, motif |
| Molecular graph | GNN, Graph Transformer | locality, bond topology, permutation equivariance |
| 3D coordinates | equivariant GNN, SE(3) Transformer, diffusion/flow | rotation/translation symmetry |
| Fingerprint | tree model, MLP, retrieval | fast sparse substructure signal |
| Pocket grid | CNN, voxel model, 3D U-Net | spatial locality on grid |
| Complex graph | heterogeneous GNN, cross-attention | pair interaction and relation edges |

## Choosing a Route

| Question | Prefer |
| --- | --- |
| molecule-only property인가? | fingerprint, molecular graph, conformer |
| target-conditioned activity인가? | ligand representation + target/pocket context |
| sequence family generalization인가? | protein sequence with family-aware split |
| pose quality가 핵심인가? | conformer, pocket, complex coordinates |
| geometry validity를 주장하는가? | coordinate representation with explicit geometry checks |
| fast screening baseline이 필요한가? | fingerprint/tree model, similarity, docking score |

## Representation Contract

각 representation note에는 아래를 남깁니다.

| Field | Write |
| --- | --- |
| Object | molecule, protein, ligand, pocket, complex, sequence, conformer |
| Preprocessing | standardization, filtering, missing atoms/residues, protonation, tautomer |
| Source | experimental, predicted, generated, docked, minimized, database-derived |
| Axes | token, atom, residue, node, edge, coordinate, candidate |
| Availability | inference time에 사용할 수 있는 정보인지 |
| Leakage risk | duplicate, homolog, scaffold, ligand-defined pocket, template, assay source |

## Split Coupling

Representation과 split은 따로 정할 수 없습니다.

| Representation | Split risk |
| --- | --- |
| Sequence | homolog, duplicate, family leakage |
| Molecule graph/fingerprint | scaffold, analog series, assay source leakage |
| Protein-ligand complex | same target, same ligand, same complex family leakage |
| Pocket/structure | ligand-defined pocket, template, predicted-vs-experimental mismatch |
| Conformer/pose | generated pose or minimized structure leaking label proxy |

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/entities|Objects and Entities]]
- [[molecular-modeling/data-evaluation|Data and Evaluation]]
- [[concepts/molecular-modeling/protein-ligand-representation-contract|Protein-ligand representation contract]]
- [[ai/model-reading-map|Model Reading Map]]
