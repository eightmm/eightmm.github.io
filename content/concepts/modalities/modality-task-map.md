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
| Text | tokens, chunks, embeddings | retrieval, similarity search, reranking, QA, generation, classification | sequence, answer, class, ranked list | context leakage, prompt injection, hallucination |
| Image | pixels, patches, feature maps, regions | classification, detection, segmentation, captioning | class, box, mask, sequence | near-duplicate leakage, augmentation mismatch |
| Video | frames, clips, tubelets, temporal tokens | event detection, captioning, retrieval, forecasting | class, segment, sequence, ranked list | temporal leakage, frame sampling bias |
| Audio | waveform, spectrogram, acoustic tokens | speech recognition, event detection, generation | sequence, class, timestamped event | sampling-rate mismatch, transcript leakage |
| Tabular | columns, normalized features, embeddings | classification, regression, ranking, reranking | class, scalar, ranked list | identifier leakage, train-test distribution shift |
| Graph | nodes, edges, features | graph classification, node prediction, link prediction | class, label, edge, graph | split leakage through shared nodes or edges |
| Sequence | tokens, residues, k-mers, events | classification, generation, tagging, similarity search, forecasting | class, sequence, span, scalar, ranked list | homolog leakage, future information leakage |
| Molecule | SMILES, graph, fingerprint, conformer | [[concepts/tasks/property-prediction|property prediction]], similarity search, generation | scalar, class, ranked list, sequence, graph | scaffold leakage, standardization mismatch |
| 3D structure | coordinates, distances, frames, surfaces, geometric graphs | pose prediction, structure prediction, scoring, similarity search, segmentation | coordinates, scalar, class, graph, ranked list | coordinate-frame leakage, template leakage |
| Protein-ligand complex | pocket graph, ligand pose, pair features, interaction graph | [[concepts/tasks/interaction-prediction|interaction prediction]], affinity prediction, pose ranking | scalar, class, contact map, ranked list | protein-family, scaffold, assay, and pose leakage |

## Task Construction

A task should be specified as:

$$
\mathcal{T}
=
(\mathcal{X}_{\mathrm{raw}}, \phi, \mathcal{Y}, v, \mathcal{L}, \mathcal{M}, s)
$$

where $\mathcal{X}_{\mathrm{raw}}$ is the raw input space, $\phi$ is preprocessing or representation, $\mathcal{Y}$ is the output space, $v$ is the validity rule, $\mathcal{L}$ is the training loss, $\mathcal{M}$ is the metric set, and $s$ is the split rule.

The metric set $\mathcal{M}$ should be chosen with [[concepts/evaluation/metric-selection|Metric selection]], and failure cases should be named with [[concepts/evaluation/failure-mode-taxonomy|Failure mode taxonomy]].

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
- What failure modes are expected and counted?
- Does the split rule prevent modality-specific leakage?
- Does the architecture match the structure of the representation rather than the name of the task?

## Related

- [[concepts/modalities/index|Modalities]]
- [[concepts/modalities/modality-representation|Modality representation]]
- [[concepts/tasks/index|Tasks]]
- [[concepts/tasks/task-specification|Task specification]]
- [[concepts/tasks/task-output-space|Task output space]]
- [[concepts/tasks/property-prediction|Property prediction]]
- [[concepts/tasks/interaction-prediction|Interaction prediction]]
- [[concepts/tasks/similarity-search|Similarity search]]
- [[concepts/tasks/reranking|Reranking]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/architectures/architecture-selection|Architecture selection]]
- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/evaluation/failure-mode-taxonomy|Failure mode taxonomy]]
- [[concepts/evaluation/leakage|Leakage]]
