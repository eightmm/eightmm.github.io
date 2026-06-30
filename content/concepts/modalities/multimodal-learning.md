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

## Multimodal Task Contract

Multimodal learning should state what each modality contributes and which modalities are available at train and inference time.

$$
\mathcal{T}
=
(\mathcal{M}_{\mathrm{train}},\ \mathcal{M}_{\mathrm{test}},\ y,\ V)
$$

| Field | Question |
| --- | --- |
| $\mathcal{M}_{\mathrm{train}}$ | which modalities are observed during training? |
| $\mathcal{M}_{\mathrm{test}}$ | which modalities are guaranteed at inference? |
| target $y$ | which modality or external label defines supervision? |
| verifier $V$ | alignment, retrieval, generation, or task metric? |

If a modality exists only at training time, the model needs distillation, missing-modality handling, or a different task definition.

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

## Fusion Patterns

| Fusion | Formula sketch | Use when |
| --- | --- | --- |
| early concatenation | $z=[z^{(1)};z^{(2)}]$ | aligned fixed-size features |
| late fusion | $\hat{y}=F(\hat{y}^{(1)},\hat{y}^{(2)})$ | independent predictors exist |
| cross-attention | $Q=z^{(a)}, K,V=z^{(b)}$ | one modality queries another |
| gated fusion | $z=g\odot z^{(a)}+(1-g)\odot z^{(b)}$ | modality reliability varies |
| shared embedding | $z^{(a)}\approx z^{(b)}$ | retrieval or alignment |

## Leakage and Shortcut Risk

| Risk | Example |
| --- | --- |
| label leakage | one modality contains target-derived metadata |
| identity shortcut | paired IDs let model memorize pairs |
| missing-modality mismatch | training uses modality absent at deployment |
| dominance | one modality carries all signal and hides failure in another |
| false negatives | contrastive batch contains unmarked matching pairs |

Evaluation should report whether the model actually uses multiple modalities or simply relies on the easiest one.

## Patterns

- Image-text alignment for retrieval, captioning, and visual question answering.
- Video-audio-text alignment for temporal understanding.
- Molecule-protein or ligand-pocket modeling for structure-based modeling.
- Text-tool-state fusion in agent systems.
- Missing-modality robustness when one signal is unavailable at deployment.

## Checks

- Are the modalities truly paired, weakly paired, or only co-occurring?
- What happens when one modality is missing or corrupted?
- Does one modality leak the label for another?
- Is the model evaluated on alignment, generation, retrieval, or downstream prediction?
- Are modality-specific failures visible in the metrics?
- Are train-time and inference-time modality sets the same?
- Does ablation show what each modality contributes?

## Related

- [[concepts/modalities/index|Modalities]]
- [[concepts/modalities/modality-alignment|Modality alignment]]
- [[concepts/modalities/missing-modality|Missing modality]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder]]
- [[concepts/sbdd/index|Structure-based drug discovery]]
