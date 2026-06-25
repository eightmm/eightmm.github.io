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

## Core Modalities

- [[concepts/modalities/text|Text]]
- [[concepts/modalities/sequence|Sequence]]
- [[concepts/modalities/image|Image]]
- [[concepts/modalities/video|Video]]
- [[concepts/modalities/audio|Audio]]
- [[concepts/modalities/tabular|Tabular]]
- [[concepts/modalities/graph|Graph]]
- [[concepts/modalities/3d-structure|3D structure]]
- [[concepts/modalities/modality-representation|Modality representation]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/modalities/modality-alignment|Modality alignment]]
- [[concepts/modalities/missing-modality|Missing modality]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]

## Common Tasks

- [[concepts/tasks/task-specification|Task specification]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/tasks/similarity-search|Similarity search]]
- [[concepts/tasks/reranking|Reranking]]
- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/tasks/segmentation|Segmentation]]
- [[concepts/tasks/captioning|Captioning]]
- [[concepts/tasks/question-answering|Question answering]]
- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/tasks/structured-prediction|Structured prediction]]

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
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/modalities/modality-alignment|Modality alignment]]
- [[concepts/modalities/missing-modality|Missing modality]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/molecular-modeling/index|Molecular modeling concepts]]
- [[concepts/protein-modeling/index|Protein modeling concepts]]
