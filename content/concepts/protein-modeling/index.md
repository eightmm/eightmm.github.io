---
title: Protein Modeling Concepts
tags:
  - protein-modeling
  - concepts
---

# Protein Modeling Concepts

Protein modeling concept는 sequence, evolutionary signal, structure, geometric constraint가 model input과 output으로 바뀌는 과정을 설명합니다.

Protein note의 기본 단위는 항상 먼저 고정합니다.

$$
u
\in
\{\text{residue}, \text{domain}, \text{chain}, \text{family}, \text{pocket}, \text{complex}\}
$$

단위가 바뀌면 representation, split, metric, leakage risk가 같이 바뀝니다.

## Route Map

| Need | Start | Risk |
| --- | --- | --- |
| choose a protein representation | [Protein representation](/concepts/protein-modeling/protein-representation) | pooling, tokenization, MSA/template leakage |
| understand sequence-only pretraining | [Protein language model](/concepts/protein-modeling/protein-language-model), [Multiple sequence alignment](/concepts/protein-modeling/multiple-sequence-alignment) | likelihood confused with fitness |
| define protein units and groups | [Protein domain](/concepts/protein-modeling/protein-domain), [Sequence identity clustering](/concepts/protein-modeling/sequence-identity-clustering) | homologs crossing train/test |
| connect sequence to structure | [Sequence-structure alignment](/concepts/protein-modeling/sequence-structure-alignment), [Residue indexing](/concepts/protein-modeling/residue-indexing) | residue mismatches, missing residues, insertion codes |
| prepare structures | [Protein structure cleaning](/concepts/protein-modeling/protein-structure-cleaning) | chain selection, alternate locations, unresolved regions |
| model structure | [Protein structure prediction](/concepts/protein-modeling/protein-structure-prediction), [Contact map](/concepts/protein-modeling/contact-map) | benchmark leakage and coordinate-frame assumptions |
| model binding context | [Binding site](/concepts/protein-modeling/binding-site), [Pocket representation](/concepts/protein-modeling/pocket-representation) | ligand-defined pockets and apo/holo mismatch |

## Task Boundary

| Task Type | Object Unit | Evidence Boundary |
| --- | --- | --- |
| sequence representation | residue, domain, chain | downstream split, pooling rule, homolog control |
| structure prediction | chain, domain, complex | template/MSA policy, coordinate target, confidence calibration |
| functional prediction | residue, domain, chain, assay row | label semantics, source, family split |
| binding-site modeling | pocket, residue set, local structure | site definition, apo/holo state, deployability |
| protein-ligand modeling | protein, ligand, pocket, complex | protein-family split plus ligand scaffold or complex split |
| design/generation | sequence, backbone, pocket, complex | validity, novelty, diversity, downstream experimental gap |

## Representation Contract

For any protein note, state what the model actually sees:

$$
r_P
=
\phi(s_{1:L}, X, c)
$$

where $s_{1:L}$ is the residue sequence, $X$ is optional structure or coordinates, and $c$ is context such as MSA, template, pocket, species, assay, ligand, or family.

| Input Source | Common Use | Main Public Claim Risk |
| --- | --- | --- |
| raw sequence | protein language models, classification, variant scoring | homolog leakage and family memorization |
| MSA | coevolution, structure prediction | database date and template leakage |
| experimental structure | geometric modeling, pockets, docking | unavailable-at-deployment information |
| predicted structure | structure-aware downstream model | circular evidence and model-bias transfer |
| pocket or ligand context | binding, docking, interaction prediction | ligand-defined pocket leakage |

## Evidence Strength

Do not read all protein metrics as the same type of evidence.

| Evidence | Supports | Does Not Prove Alone |
| --- | --- | --- |
| sequence likelihood | sequence distribution or representation quality | binding, function, expression, foldability |
| contact/distance accuracy | coarse structural constraints | ligand-ready all-atom pocket geometry |
| backbone accuracy | fold-level structure quality | side-chain placement or docking quality |
| pocket localization | site-finding under a stated definition | affinity or pose scoring |
| downstream benchmark score | task-specific performance | broad biological generalization |

## Core Concepts

| Group | Notes |
| --- | --- |
| Representation | [Protein representation](/concepts/protein-modeling/protein-representation), [Protein language model](/concepts/protein-modeling/protein-language-model), [Multiple sequence alignment](/concepts/protein-modeling/multiple-sequence-alignment) |
| Structure | [Protein structure cleaning](/concepts/protein-modeling/protein-structure-cleaning), [Protein structure prediction](/concepts/protein-modeling/protein-structure-prediction), [Contact map](/concepts/protein-modeling/contact-map) |
| Indexing | [Residue indexing](/concepts/protein-modeling/residue-indexing), [Sequence-structure alignment](/concepts/protein-modeling/sequence-structure-alignment) |
| Binding | [Binding site](/concepts/protein-modeling/binding-site), [Pocket representation](/concepts/protein-modeling/pocket-representation) |

## Checks

- What is the example unit: residue, domain, chain, family, pocket, complex, or assay row?
- Is the model sequence-only, structure-only, or sequence-structure fused?
- What is available at inference time: MSA, template, predicted structure, experimental structure, ligand, or pocket?
- Is sequence likelihood being treated as evidence for function, stability, or binding without downstream validation?
- Is the pocket representation ligand-defined, predicted, or deployable?
- Are residue indices aligned across sequence tokens, structure residues, and coordinates?
- Are homologs and near-duplicate proteins separated across splits?
- Are missing residues, insertion codes, non-standard residues, and chain IDs handled explicitly?
- Does the evaluation measure geometry, function, interaction, or downstream task transfer?
- Is the claim narrowed to the evidence type: sequence, structure, pocket, binding, or function?

## Related

- [[molecular-modeling/protein-modeling|Protein modeling]]
- [[entities/protein|Protein]]
- [[entities/sequence|Sequence]]
- [[entities/structure|Structure]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[concepts/protein-modeling/protein-language-model|Protein language model]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
