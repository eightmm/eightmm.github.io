---
title: Learning Phrase Representations using RNN Encoder-Decoder
aliases:
  - papers/gru
  - papers/rnn-encoder-decoder
tags:
  - papers
  - architectures
  - recurrent
---

# Learning Phrase Representations using RNN Encoder-Decoder

> The paper introduced a gated recurrent encoder-decoder model and popularized the GRU-style recurrent unit.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation |
| Authors | Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio |
| Year | 2014 |
| Venue | EMNLP 2014 |
| arXiv | [1406.1078](https://arxiv.org/abs/1406.1078) |
| ACL Anthology | [D14-1179](https://aclanthology.org/D14-1179/) |
| Status | verified |

## Question

Phrase-based statistical machine translation needed better learned phrase representations and phrase-pair scores. The paper asked whether an encoder-decoder recurrent network could map variable-length source phrases into representations useful for translation.

## Main Claim

An RNN encoder-decoder with gated hidden-state updates can learn phrase representations that improve translation scoring.

GRU-style update:

$$
\begin{aligned}
z_t &= \sigma(W_z x_t + U_z h_{t-1}) \\
r_t &= \sigma(W_r x_t + U_r h_{t-1}) \\
\tilde{h}_t &= \tanh(W_h x_t + U_h(r_t \odot h_{t-1})) \\
h_t &= (1-z_t)\odot h_{t-1} + z_t \odot \tilde{h}_t
\end{aligned}
$$

The update gate $z_t$ interpolates between keeping the old state and writing a new candidate state.

## Method

| Component | Role |
| --- | --- |
| encoder RNN | compresses source phrase into a vector |
| decoder RNN | generates target phrase conditioned on the vector |
| reset gate | controls how much past state enters the candidate update |
| update gate | controls state replacement |
| conditional probability | scores phrase-pair compatibility |

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Encoder-decoder scores improve phrase-based SMT | translation system experiments | tied to pre-neural SMT pipelines |
| Gated recurrence learns meaningful phrase embeddings | qualitative embedding analysis | representation quality depends on training data and objective |
| Variable-length inputs can be handled by recurrent encoding | phrase modeling setup | fixed vector bottleneck became a limitation for long sequences |

## Limitations

- The fixed-length encoder vector is a bottleneck for long or detailed sequences.
- Attention-based encoder-decoder models quickly became stronger for translation.
- GRU and LSTM differ in state structure, gating, and empirical behavior.
- The paper is both an architecture and NLP-system paper; the architecture claim should be separated from SMT-specific evidence.

## Why It Matters

This is the key paper for GRU-style gated recurrence and for reading encoder-decoder architectures before attention became standard.

## Connections

- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/gru|GRU]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder]]
- [[papers/architectures/long-short-term-memory|Long Short-Term Memory]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/index|Architecture papers]]
