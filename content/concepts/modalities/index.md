---
title: Modalities
tags:
  - modalities
  - machine-learning
---

# Modalities

Modality는 고유한 structure, noise pattern, inductive bias를 가진 input 또는 output signal의 종류입니다. Text, image, video, audio, molecule, protein, graph, 3D structure는 downstream task가 같은 prediction problem처럼 쓰여도 서로 다른 representation을 요구하는 경우가 많습니다.

Model input은 modality-specific view의 묶음으로 볼 수 있습니다.

$$
x = \{x^{(m)}\}_{m \in \mathcal{M}}
$$

where $m$ indexes a modality and $\mathcal{M}$ is the set of available modalities.

## Route Map

| Question | Start | Then Check |
| --- | --- | --- |
| what is the raw signal? | [Text](/concepts/modalities/text), [Sequence](/concepts/modalities/sequence), [Image](/concepts/modalities/image), [Video](/concepts/modalities/video), [Audio](/concepts/modalities/audio), [Tabular](/concepts/modalities/tabular) | preprocessing and lost information |
| is the object relational? | [Graph](/concepts/modalities/graph) | node/edge definition and graph construction |
| is the object geometric? | [3D structure](/concepts/modalities/3d-structure) | coordinate frame, invariance, equivariance |
| what does the model actually see? | [Modality representation](/concepts/modalities/modality-representation), [Representation contract](/concepts/modalities/representation-contract) | tensor shape, tokens, graph, coordinates |
| how does modality map to task? | [Modality-task map](/concepts/modalities/modality-task-map), [Task specification](/concepts/tasks/task-specification) | output space, loss, metric, split |
| are multiple modalities aligned? | [Modality alignment](/concepts/modalities/modality-alignment), [Multimodal learning](/concepts/modalities/multimodal-learning) | missing modality and leakage |

## Modality Contract

Modality note should separate the raw signal from the model-ready representation.

$$
x_{\mathrm{raw}}^{(m)}
\xrightarrow{\phi_m}
r^{(m)}
\xrightarrow{f_\theta}
\hat{y}
$$

Here $x_{\mathrm{raw}}^{(m)}$ is the raw signal, $\phi_m$ is modality-specific preprocessing or featurization, and $r^{(m)}$ is what the model actually sees.

| Layer | Ask | Route |
| --- | --- | --- |
| raw signal | text, sequence, image, graph, coordinates, tabular record, audio, video? | modality-specific pages |
| representation | token, patch, feature map, graph, coordinate set, embedding, fingerprint? | [Representation contract](/concepts/modalities/representation-contract) |
| axes | batch, token, channel, node, edge, residue, atom, frame, coordinate? | [Tensor shape notation](/concepts/math/tensor-shape-notation) |
| lost information | what is discarded by tokenization, cropping, graph construction, or featurization? | [Preprocessing contract](/concepts/data/preprocessing-contract) |
| output behavior | class, scalar, rank, sequence, graph, coordinate, action? | [Task output space](/concepts/tasks/task-output-space) |
| evaluation risk | what leakage or metric failure is modality-specific? | [Evaluation](/concepts/evaluation) |

## Modality vs Entity vs Task

These axes often get mixed.

| Axis | Meaning | Example |
| --- | --- | --- |
| entity | real-world object being modeled | molecule, protein, ligand, pocket, document, image |
| modality | signal form available to the model | sequence, graph, image, 3D coordinates, text |
| representation | processed input object | SMILES tokens, molecular graph, residue embedding, voxel grid |
| task | target behavior | classify, rank, retrieve, generate, predict coordinates |
| architecture | computation pattern | Transformer, CNN, GNN, SSM, equivariant GNN |

For example, a protein can appear as sequence tokens, an MSA, a residue graph, a contact map, 3D coordinates, or a text description. Those are different modality/representation choices for the same entity.

## Representation Choice Map

| Raw object | Possible representations | Common architecture route |
| --- | --- | --- |
| text document | tokens, chunks, embeddings | Transformer, retrieval encoder |
| protein sequence | residues, k-mers, PLM embeddings, MSA | Transformer, SSM, CNN, RNN |
| molecule | SMILES, graph, fingerprint, descriptors, conformer | Transformer, GNN, MLP, equivariant model |
| protein-ligand complex | pocket graph, pair graph, distance features, coordinates | GNN, graph transformer, equivariant GNN |
| image | pixels, patches, feature maps, regions | CNN, ViT, U-Net |
| video | frames, clips, tubelets, temporal tokens | CNN/ViT plus temporal model |
| tabular record | columns, normalized features, learned embeddings | linear model, tree model, MLP |
| 3D structure | coordinates, distances, frames, surfaces, voxels | equivariant model, GNN, CNN |

The right note for a representation claim is usually [[concepts/modalities/representation-contract|Representation contract]], not only the architecture page.

## Modality Groups

| Group | Notes | Common Tasks |
| --- | --- | --- |
| Language and symbolic sequence | [Text](/concepts/modalities/text), [Sequence](/concepts/modalities/sequence) | [Question answering](/concepts/tasks/question-answering), [Sequence generation](/concepts/tasks/sequence-generation), [Retrieval](/concepts/tasks/retrieval) |
| Dense sensory input | [Image](/concepts/modalities/image), [Video](/concepts/modalities/video), [Audio](/concepts/modalities/audio) | [Object detection](/concepts/tasks/object-detection), [Segmentation](/concepts/tasks/segmentation), [Captioning](/concepts/tasks/captioning) |
| Structured objects | [Tabular](/concepts/modalities/tabular), [Graph](/concepts/modalities/graph), [3D structure](/concepts/modalities/3d-structure) | [Structured prediction](/concepts/tasks/structured-prediction), [Graph prediction](/concepts/tasks/graph-prediction), [Coordinate prediction](/concepts/tasks/coordinate-prediction) |
| Cross-modal systems | [Modality alignment](/concepts/modalities/modality-alignment), [Missing modality](/concepts/modalities/missing-modality), [Multimodal learning](/concepts/modalities/multimodal-learning) | retrieval, grounding, fusion, reranking |

## Failure Modes

| Failure | Example |
| --- | --- |
| raw and processed object confused | evaluating a molecule graph task as if it were a SMILES task |
| axis meaning omitted | node axis, token axis, channel axis, or coordinate axis unclear |
| representation leakage | graph, crop, pocket, template, or retrieval context uses test-only information |
| alignment leakage | paired modalities share source or near-duplicate examples across splits |
| modality missingness ignored | model assumes image, sequence, structure, or metadata always exists |
| metric mismatch | generated sequence validity treated like classification accuracy |
| architecture keyword overused | “Transformer” claim hides tokenization, context, or task differences |

## Why It Matters

- The raw signal and the model-ready representation are often different objects.
- Tokenization can discard timing, geometry, spatial locality, or alignment information.
- Single-modal, cross-modal, and multimodal-fusion tasks fail in different ways.
- Evaluation metrics must match the output modality, not only the model family.
- A modality should be connected to task output space, loss, metric, and split before choosing an architecture.

## Public Note Pattern

| Field | Write |
| --- | --- |
| raw signal | what exists before preprocessing |
| representation | what tensor/token/graph/coordinate object the model sees |
| axes | shape and axis names |
| preprocessing | deterministic, learned, stochastic, cached, or versioned |
| task | output space and valid output |
| metric | primary metric and invalid-output handling |
| split | leakage-relevant unit |
| boundary | whether entity-specific details belong in Computational Biology |

## Checks

- What is the raw input signal?
- What is the tensor, token, graph, or coordinate representation after preprocessing?
- What [[concepts/modalities/modality-representation|modality representation]] does the model actually see?
- What [[concepts/modalities/representation-contract|representation contract]] connects raw object, model input, output, loss, metric, split, and leakage?
- What [[concepts/modalities/modality-task-map|modality-task map]] connects input, output, loss, metric, and split?
- What [[concepts/data/preprocessing-contract|preprocessing contract]] turns the raw signal into model input?
- Which information is lost before the model sees the input?
- Does the model need alignment between modalities?
- Can the model handle missing, delayed, or corrupted modalities?
- Are splits designed to prevent near-duplicate or cross-modal leakage?

## Related

- [[ai/index|AI]]
- [[concepts/tasks/index|Tasks]]
- [[concepts/architectures/index|Architectures]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/modalities/modality-representation|Modality representation]]
- [[concepts/modalities/representation-contract|Representation contract]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/modalities/modality-alignment|Modality alignment]]
- [[concepts/modalities/missing-modality|Missing modality]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/molecular-modeling/index|Molecular modeling concepts]]
- [[concepts/protein-modeling/index|Protein modeling concepts]]
