---
title: Modality Representation
tags:
  - modalities
  - preprocessing
  - representation-learning
---

# Modality Representation

A modality representation is the model-ready form of a raw signal. The raw object may be text, image, video, audio, graph, molecule, protein sequence, or 3D coordinates; the model sees tokens, tensors, graphs, embeddings, or coordinate features.

The representation map is:

$$
r = \phi(x_{\mathrm{raw}})
$$

where $x_{\mathrm{raw}}$ is the original signal, $\phi$ is the preprocessing or encoder function, and $r$ is the model input.

For multimodal input:

$$
r^{(m)} = \phi_m(x^{(m)}),
\qquad
r = F(r^{(1)},\ldots,r^{(M)})
$$

where $m$ indexes modalities and $F$ is fusion, alignment, routing, or concatenation.

## Common Forms

- Text: string to tokens, embeddings, chunks, or retrieved passages.
- Image: pixels to patches, feature maps, regions, masks, or visual embeddings.
- Video: frames to clips, tubelets, event segments, or temporal tokens.
- Audio: waveform to samples, spectrograms, acoustic tokens, or transcript text.
- Graph: raw entities to nodes, edges, node features, and edge features.
- 3D structure: coordinates to distances, frames, surfaces, voxel grids, geometric graphs, or invariant/equivariant features.

## Information Contract

Every representation chooses what to preserve and what to discard:

$$
x_{\mathrm{raw}}
\xrightarrow{\phi}
r
\xrightarrow{f_\theta}
\hat{y}
$$

If $\phi$ removes information needed for the task, the model cannot recover it. If $\phi$ includes deployment-unavailable information, evaluation can become invalid.

Representation should be chosen together with [[concepts/modalities/modality-task-map|Modality-task map]], because the same raw signal may support classification, retrieval, generation, localization, ranking, or structured prediction.

## Checks

- What is the raw signal, and what exactly is the model input?
- Is $\phi$ deterministic, learned, stochastic, cached, or data-dependent?
- Which information is discarded: order, time, geometry, resolution, units, metadata, uncertainty, or missingness?
- Does $\phi$ use labels, future information, target poses, test-set statistics, or private metadata?
- Are train, validation, test, and inference preprocessing contracts identical where they should be?
- Is the representation stable under expected noise, corruption, and missing modalities?

## Leakage Risks

- Cropping or selecting regions using ground-truth targets unavailable at deployment.
- Computing normalization or vocabulary statistics using test data.
- Using metadata that encodes labels or source identity.
- Splitting after representation caching, causing near-duplicate leakage.
- Aligning modalities using future timestamps or evaluation-only annotations.

## Related

- [[concepts/modalities/index|Modalities]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]]
- [[concepts/modalities/modality-alignment|Modality alignment]]
- [[concepts/modalities/missing-modality|Missing modality]]
- [[concepts/evaluation/leakage|Leakage]]
