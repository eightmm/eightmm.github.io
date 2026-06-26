---
title: Modalities
tags:
  - modalities
  - machine-learning
---

# Modalities

A modality is a type of input or output signal with its own structure, noise pattern, and inductive bias. Text, images, video, audio, molecules, proteins, graphs, and 3D structures often require different representations even when the downstream task is written as the same prediction problem.

A model input can be viewed as a collection of modality-specific views:

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

## Modality Groups

| Group | Notes | Common Tasks |
| --- | --- | --- |
| Language and symbolic sequence | [Text](/concepts/modalities/text), [Sequence](/concepts/modalities/sequence) | [Question answering](/concepts/tasks/question-answering), [Sequence generation](/concepts/tasks/sequence-generation), [Retrieval](/concepts/tasks/retrieval) |
| Dense sensory input | [Image](/concepts/modalities/image), [Video](/concepts/modalities/video), [Audio](/concepts/modalities/audio) | [Object detection](/concepts/tasks/object-detection), [Segmentation](/concepts/tasks/segmentation), [Captioning](/concepts/tasks/captioning) |
| Structured objects | [Tabular](/concepts/modalities/tabular), [Graph](/concepts/modalities/graph), [3D structure](/concepts/modalities/3d-structure) | [Structured prediction](/concepts/tasks/structured-prediction), [Graph prediction](/concepts/tasks/graph-prediction), [Coordinate prediction](/concepts/tasks/coordinate-prediction) |
| Cross-modal systems | [Modality alignment](/concepts/modalities/modality-alignment), [Missing modality](/concepts/modalities/missing-modality), [Multimodal learning](/concepts/modalities/multimodal-learning) | retrieval, grounding, fusion, reranking |

## Why It Matters

- The raw signal and the model-ready representation are often different objects.
- Tokenization can discard timing, geometry, spatial locality, or alignment information.
- Single-modal, cross-modal, and multimodal-fusion tasks fail in different ways.
- Evaluation metrics must match the output modality, not only the model family.
- A modality should be connected to task output space, loss, metric, and split before choosing an architecture.

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
