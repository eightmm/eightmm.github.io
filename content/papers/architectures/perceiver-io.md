---
title: Perceiver IO
aliases:
  - papers/perceiver-io
tags:
  - papers
  - architectures
  - attention
  - multimodal
---

# Perceiver IO

> The paper generalized the Perceiver latent-bottleneck idea to flexible structured inputs and outputs.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Perceiver IO: A General Architecture for Structured Inputs & Outputs |
| Authors | Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier Henaff, Matthew M. Botvinick, Andrew Zisserman, Oriol Vinyals, Joao Carreira |
| Year | 2021 |
| Venue | ICLR 2022 |
| arXiv | [2107.14795](https://arxiv.org/abs/2107.14795) |
| OpenReview | [fILj7WpI-g](https://openreview.net/forum?id=fILj7WpI-g) |
| Status | verified |

## Question

Transformers are flexible but scale poorly with very large input or output arrays. Domain-specific architectures scale better but bake in modality assumptions. The question was whether one architecture could handle arbitrary input and output structures with a shared latent processing core.

## Main Claim

Perceiver IO uses cross-attention to map large inputs into a latent array, processes the latent array, then uses output queries to decode structured outputs.

Input cross-attention:

$$
Z'
=
\operatorname{softmax}
\left(
\frac{Q_Z K_X^\top}{\sqrt{d}}
\right)
V_X
$$

Output query decoding:

$$
Y
=
\operatorname{Attn}(Q_Y, K_Z, V_Z)
$$

The expensive self-attention happens mostly in a fixed-size latent space rather than over all input positions.

## Method

| Component | Role |
| --- | --- |
| input adapters | encode raw modality-specific inputs |
| latent array | fixed-size bottleneck for iterative processing |
| cross-attention | moves information between input/output arrays and latents |
| latent self-attention | performs main computation at bounded cost |
| output queries | specify desired output structure |

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| One architecture can cover many domains | language, vision, multimodal, optical flow, and game tasks | adapters and task formatting still matter |
| Latent bottlenecks scale to large inputs | comparison with direct Transformer attention | bottleneck size becomes a modeling choice |
| Flexible output queries avoid task-specific heads | structured output experiments | query design can encode task assumptions |

## Limitations

- "General architecture" does not remove the need for input encoding, output query design, and task-specific losses.
- Latent bottlenecks can lose information if too small or poorly queried.
- Strong domain-specialized backbones can still outperform it under matched compute.
- The architecture is more complex to reason about than a standard encoder-only or decoder-only Transformer.

## Why It Matters

Perceiver IO is a useful bridge between Transformers, multimodal models, set processing, and large structured input/output problems.

## Connections

- [[concepts/architectures/perceiver|Perceiver]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/set-transformer|Set Transformer]]
- [[papers/architectures/set-transformer|Set Transformer]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/index|Architecture papers]]
