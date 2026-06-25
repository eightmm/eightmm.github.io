---
title: Multimodal Learning
tags:
  - modalities
  - multimodal
  - representation-learning
---

# Multimodal Learning

Multimodal learning uses more than one modality in the same task. The central question is whether the model should align, fuse, retrieve across, or generate between modalities.

Each modality can be encoded separately:

$$
z^{(m)} = f_\theta^{(m)}(x^{(m)})
$$

Fusion combines modality-specific embeddings:

$$
z = F_\phi(z^{(1)},\ldots,z^{(M)})
$$

where $F_\phi$ can be concatenation, pooling, cross-attention, gated fusion, or a larger joint model.

## Alignment Objective

A common contrastive alignment loss between modalities $a$ and $b$ is:

$$
\mathcal{L}_i =
-\log
\frac{
\exp(\operatorname{sim}(z_i^{(a)}, z_i^{(b)})/\tau)
}{
\sum_j \exp(\operatorname{sim}(z_i^{(a)}, z_j^{(b)})/\tau)
}
$$

where $\operatorname{sim}$ is a similarity function and $\tau$ is a temperature.

## Patterns

- Image-text alignment for retrieval, captioning, and visual question answering.
- Video-audio-text alignment for temporal understanding.
- Molecule-protein or ligand-pocket modeling for structure-based AI.
- Text-tool-state fusion in agent systems.
- Missing-modality robustness when one signal is unavailable at deployment.

## Checks

- Are the modalities truly paired, weakly paired, or only co-occurring?
- What happens when one modality is missing or corrupted?
- Does one modality leak the label for another?
- Is the model evaluated on alignment, generation, retrieval, or downstream prediction?
- Are modality-specific failures visible in the metrics?

## Related

- [[concepts/modalities/index|Modalities]]
- [[concepts/modalities/modality-alignment|Modality alignment]]
- [[concepts/modalities/missing-modality|Missing modality]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder]]
- [[concepts/sbdd/index|Structure-based drug discovery]]
