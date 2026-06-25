---
title: Genome Sequence Modeling
tags:
  - research
  - genome
  - sequence-modeling
---

# Genome Sequence Modeling

Genome sequence modeling is a narrow bio-AI scope for this site: sequence representations, long-context architectures, masked objectives, and evaluation issues around genomic data.

A sequence model usually estimates token probabilities or representations:

$$
p_\theta(x_t \mid x_{<t})
\quad\text{or}\quad
h_t = f_\theta(x_{1:T})_t
$$

The long-context question is how much of $x_{1:T}$ can influence $h_t$ or the prediction at $t$.

## In Scope

- Genome and DNA sequence representation.
- Long-context [[concepts/architectures/transformer|Transformer]] and [[concepts/architectures/state-space-model|state-space model]] architectures.
- [[concepts/learning/self-supervised-learning|Self-supervised learning]] and [[concepts/learning/masked-modeling|masked modeling]] objectives.
- Leakage, split design, and out-of-distribution evaluation.

## Out of Scope for Now

- Single-cell analysis.
- Transcriptomics.
- Broad omics integration.
- Pathway or systems biology.
- Clinical interpretation workflows.

## Related

- [[entities/genome|Genome]]
- [[entities/sequence|Sequence]]
- [[concepts/architectures/mamba|Mamba]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
