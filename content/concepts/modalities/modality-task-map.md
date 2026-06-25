---
title: Modality-Task Map
tags:
  - modalities
  - tasks
  - evaluation
---

# Modality-Task Map

A modality-task map connects the raw input signal to the model-ready representation, target output, loss, metric, and evaluation risk. It prevents a note from saying only "use a Transformer" or "use a GNN" without specifying what the model must actually solve.

The full contract is:

$$
x_{\mathrm{raw}}
\xrightarrow{\phi}
r
\xrightarrow{f_\theta}
\hat{y}
\in
\mathcal{Y}
$$

where $x_{\mathrm{raw}}$ is the raw modality, $\phi$ is the representation map, $r$ is the model input, $f_\theta$ is the model, and $\mathcal{Y}$ is the [[concepts/tasks/task-output-space|task output space]].

## Mapping Axes

| Modality | Common representation | Common tasks | Output space | Typical risks |
| --- | --- | --- | --- | --- |
| Text | tokens, chunks, embeddings | retrieval, QA, generation, classification | sequence, answer, class, ranked list | context leakage, prompt injection, hallucination |
| Image | pixels, patches, feature maps, regions | classification, detection, segmentation, captioning | class, box, mask, sequence | near-duplicate leakage, augmentation mismatch |
| Video | frames, clips, tubelets, temporal tokens | event detection, captioning, retrieval, forecasting | class, segment, sequence, ranked list | temporal leakage, frame sampling bias |
| Audio | waveform, spectrogram, acoustic tokens | speech recognition, event detection, generation | sequence, class, timestamped event | sampling-rate mismatch, transcript leakage |
| Tabular | columns, normalized features, embeddings | classification, regression, ranking | class, scalar, ranked list | identifier leakage, train-test distribution shift |
| Graph | nodes, edges, features | graph classification, node prediction, link prediction | class, label, edge, graph | split leakage through shared nodes or edges |
| Sequence | tokens, residues, k-mers, events | classification, generation, tagging, forecasting | class, sequence, span, scalar | homolog leakage, future information leakage |
| 3D structure | coordinates, distances, frames, surfaces, geometric graphs | pose prediction, structure prediction, scoring, segmentation | coordinates, scalar, class, graph | coordinate-frame leakage, template leakage |

## Task Construction

A task should be specified as:

$$
\mathcal{T}
=
(\mathcal{X}_{\mathrm{raw}}, \phi, \mathcal{Y}, v, \mathcal{L}, \mathcal{M}, s)
$$

where $\mathcal{X}_{\mathrm{raw}}$ is the raw input space, $\phi$ is preprocessing or representation, $\mathcal{Y}$ is the output space, $v$ is the validity rule, $\mathcal{L}$ is the training loss, $\mathcal{M}$ is the metric set, and $s$ is the split rule.

## Evaluation Implications

The same modality can support very different metrics:

$$
(\text{modality}, \mathcal{Y})
\Rightarrow
(\mathcal{L}, \mathcal{M}, \text{failure mode})
$$

For example, an image can produce a class, box, mask, caption, embedding, or retrieved item. A protein-ligand complex can produce a pose, score, affinity estimate, contact map, or ranked candidate list. These outputs require different validity checks and metrics.

## Checks

- What is the raw modality before preprocessing?
- What representation does the model actually see?
- What is one valid target output?
- What outputs are invalid and how are they handled?
- Does the loss match the output space?
- Does the metric match the user-facing behavior?
- Does the split rule prevent modality-specific leakage?
- Does the architecture match the structure of the representation rather than the name of the task?

## Related

- [[concepts/modalities/index|Modalities]]
- [[concepts/modalities/modality-representation|Modality representation]]
- [[concepts/tasks/index|Tasks]]
- [[concepts/tasks/task-specification|Task specification]]
- [[concepts/tasks/task-output-space|Task output space]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/architectures/architecture-selection|Architecture selection]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/leakage|Leakage]]
