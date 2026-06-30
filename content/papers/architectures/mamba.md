---
title: Mamba
aliases:
  - papers/mamba
  - papers/selective-state-space-models
tags:
  - papers
  - architectures
  - state-space-model
  - sequence-modeling
---

# Mamba

> The paper introduced selective state-space models as a sequence-modeling backbone with input-dependent dynamics and linear-time scaling.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Mamba: Linear-Time Sequence Modeling with Selective State Spaces |
| Authors | Albert Gu, Tri Dao |
| Year | 2023 preprint; 2024 conference |
| Venue | COLM 2024 |
| arXiv | [2312.00752](https://arxiv.org/abs/2312.00752) |
| OpenReview | [tEYskw1VY2](https://openreview.net/forum?id=tEYskw1VY2) |
| Status | verified |

## Question

Transformers mix tokens with attention, but dense attention scales quadratically in sequence length. Previous subquadratic sequence models often struggled on information-dense discrete modalities. The question was whether a state-space sequence model could add content-dependent selection while keeping linear-time scaling.

## Main Claim

Selective state-space models make SSM parameters input-dependent, allowing sequence models to choose what to propagate or forget while retaining efficient long-sequence computation.

Narrowed claim:

$$
h_t
=
\bar{A}(x_t)h_{t-1}
+
\bar{B}(x_t)x_t
$$

$$
y_t
=
C(x_t)h_t
$$

where the transition and readout depend on the current input token.

## Method

Mamba combines:

| Component | Role |
| --- | --- |
| selective SSM | input-dependent recurrence over sequence length |
| hardware-aware scan | parallelizes recurrent computation efficiently |
| gated block design | builds a simple sequence backbone around the selective SSM |
| linear sequence scaling | avoids dense $T^2$ attention cost |

The architecture changes the sequence mixing primitive from dense pairwise attention to recurrent state updates with selective dynamics.

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Selective SSMs improve prior SSM weakness on discrete data | language modeling and synthetic/selective-copy style tasks | results depend on implementation and training setup |
| Mamba scales linearly with sequence length | algorithmic design and long-sequence experiments | hardware kernels are part of the practical claim |
| Mamba is competitive across modalities | language, audio, and genomics experiments | not a proof that attention is obsolete |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task | sequence modeling |
| Input/output unit | sequence tokens to sequence representations or predictions |
| Main comparison | Transformer and prior subquadratic sequence models |
| Main metric family | language-modeling loss, downstream accuracy, throughput, sequence-length scaling |
| Not directly tested | all Transformer workloads, tool-use agents, structure-based molecular modeling |

## Limitations

- The paper's strongest practical claims depend on custom kernels and hardware-aware implementation.
- Linear-time scaling does not automatically imply better quality at every compute budget.
- Input-dependent recurrence changes interpretability and memory behavior relative to attention.
- Later Mamba variants may change the best block design, normalization, hybridization, and scaling behavior.

## Why It Matters

Mamba is a key anchor for modern state-space model architectures and alternatives to attention-based token mixing.

## Connections

- [[concepts/architectures/mamba|Mamba]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[papers/architectures/index|Architecture papers]]
