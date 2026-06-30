---
title: Modality Alignment
tags:
  - modalities
  - multimodal
  - alignment
---

# Modality Alignment

Modality alignment connects signals from different modalities that refer to the same object, event, region, or concept. It is the bridge between text-image retrieval, video-audio synchronization, molecule-protein interaction, and agent state-tool traces.

Given paired examples $(x_i^{(a)}, x_i^{(b)})$, encoders produce embeddings:

$$
z_i^{(a)} = f_a(x_i^{(a)}),
\qquad
z_i^{(b)} = f_b(x_i^{(b)})
$$

The goal is to make matched pairs more similar than mismatched pairs.

## Alignment Unit

Alignment depends on what counts as a pair.

$$
a_i \leftrightarrow b_i
$$

can mean document-level, object-level, region-level, token-level, temporal, or structural correspondence.

| Alignment unit | Example |
| --- | --- |
| global | image-caption, paper-abstract |
| local | word-region, residue-atom, ligand-pocket contact |
| temporal | audio frame-video frame |
| structural | sequence-position to coordinate/residue |
| weak pair | co-occurring text and object without exact correspondence |

The alignment unit should match the downstream task. Global alignment may be insufficient for local reasoning.

## Contrastive Alignment

A common bidirectional contrastive loss is:

$$
\mathcal{L}_{a\to b}
=
-\frac{1}{N}\sum_{i=1}^{N}
\log
\frac{
\exp(\operatorname{sim}(z_i^{(a)}, z_i^{(b)})/\tau)
}{
\sum_{j=1}^{N}
\exp(\operatorname{sim}(z_i^{(a)}, z_j^{(b)})/\tau)
}
$$

$$
\mathcal{L}
=
\frac{1}{2}
\left(
\mathcal{L}_{a\to b}
+
\mathcal{L}_{b\to a}
\right)
$$

where $\operatorname{sim}$ is a similarity score, $\tau$ is a temperature, and $N$ is batch size.

## Negative Set Problem

Contrastive alignment assumes other items in the batch are negatives:

$$
(x_i^{(a)}, x_j^{(b)}),\quad j\ne i
$$

But in real datasets, some of these may be unlabeled positives or near-duplicates.

| Problem | Consequence |
| --- | --- |
| duplicate captions or documents | model is penalized for matching valid alternatives |
| protein or molecule families | related entities treated as negatives |
| many-to-many relations | one object has several correct partners |
| batch/source artifacts | negatives differ by source rather than semantic mismatch |

This is especially important in scientific data where similarity can be graded rather than binary.

## Evaluation

| Claim | Metric or check |
| --- | --- |
| global retrieval works | Recall@k, MRR, nDCG in both directions |
| local alignment works | localization, contact, span, or correspondence metric |
| representation transfers | downstream task under frozen or fine-tuned encoder |
| alignment is robust | missing-modality or domain-shift evaluation |
| no identity shortcut | grouped split by entity/source |

## Alignment Types

- Global alignment: one caption to one image, one abstract to one paper, one ligand to one target.
- Local alignment: words to regions, frames to audio spans, residues to coordinates.
- Temporal alignment: events matched by time.
- Structural alignment: sequences, graphs, or coordinates matched by correspondence.
- Weak alignment: co-occurrence without exact local labels.

## Checks

- Are pairs truly matched, weakly paired, or only co-occurring?
- Are negatives valid, or can they include unmarked positives?
- Is alignment global, local, temporal, or structural?
- Does one modality leak the label or identity of the other?
- Is retrieval evaluated in both directions?
- Are near-duplicates or many-to-many positives handled in the negative set?
- Is the alignment unit local enough for the downstream claim?

## Related

- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[concepts/modalities/missing-modality|Missing modality]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/sbdd/interaction-fingerprint|Interaction fingerprint]]
