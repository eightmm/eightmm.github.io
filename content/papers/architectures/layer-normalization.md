---
title: Layer Normalization
aliases:
  - papers/layer-normalization
  - papers/layernorm
tags:
  - papers
  - architectures
  - normalization
---

# Layer Normalization

> The paper introduced normalization over hidden units within a single example, avoiding dependence on mini-batch statistics.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Layer Normalization |
| Authors | Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton |
| Year | 2016 |
| Venue | arXiv preprint |
| arXiv | [1607.06450](https://arxiv.org/abs/1607.06450) |
| PDF | [author PDF](https://www.cs.toronto.edu/~hinton/absps/LayerNormalization.pdf) |
| Status | verified |

## Question

BatchNorm depends on mini-batch statistics and is awkward for recurrent computation. The question was whether normalization could stabilize hidden activations using statistics from a single training example instead.

## Main Claim

LayerNorm normalizes across hidden units within a layer for each example, making the computation independent of batch size and consistent between training and inference.

Narrowed claim:

$$
\mu
=
\frac{1}{H}
\sum_{i=1}^{H} a_i
$$

$$
\sigma
=
\sqrt{
\frac{1}{H}
\sum_{i=1}^{H}
(a_i-\mu)^2
}
$$

$$
y_i
=
\gamma_i
\frac{a_i-\mu}{\sigma+\epsilon}
+
\beta_i
$$

where statistics are computed over hidden units of the same example.

## Method

LayerNorm changes the axis of normalization:

| Normalization | Statistics over | Train/test behavior |
| --- | --- | --- |
| BatchNorm | mini-batch examples | uses batch stats in training and running stats in inference |
| LayerNorm | hidden units in one example | same computation in training and inference |

This makes LayerNorm natural for recurrent networks and later Transformer blocks.

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| LayerNorm stabilizes recurrent hidden dynamics | recurrent-network experiments | original paper predates modern Transformer scaling |
| LayerNorm does not depend on batch size | method definition and empirical behavior | normalization axis still affects representation |
| LayerNorm can reduce training time | comparisons against baseline training | later architectures changed placement and residual design |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task | stabilizing neural network training |
| Architecture family | normalization layer |
| Main comparison | batch normalization and unnormalized networks |
| Main metric | training speed and task performance |
| Not directly tested | modern pre-norm Transformers at scale |

## Limitations

- LayerNorm is not universally better than BatchNorm; CNNs and Transformers often prefer different normalization choices.
- The axis being normalized matters for representation and gradient flow.
- Later Transformer practice changed where LayerNorm is placed relative to residual branches.
- The original evidence is smaller than modern large-model settings.

## Why It Matters

LayerNorm became a default component of Transformer-style architectures and is central to understanding pre-norm, post-norm, residual depth, and sequence-model stability.

## Connections

- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/normalization-placement|Normalization placement]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/rnn|RNN]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/index|Architecture papers]]
