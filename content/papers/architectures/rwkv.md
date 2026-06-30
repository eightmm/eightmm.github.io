---
title: RWKV
aliases:
  - papers/rwkv
  - papers/reinventing-rnns-for-the-transformer-era
tags:
  - papers
  - architectures
  - recurrent
  - sequence-modeling
  - language-model
---

# RWKV

> The paper presents Receptance Weighted Key Value as a language-model architecture that can be trained in a Transformer-like parallel form and run in an RNN-like recurrent form.

## Metadata

| Field | Value |
| --- | --- |
| Paper | RWKV: Reinventing RNNs for the Transformer Era |
| Authors | Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Jiaju Lin, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Guangyu Song, Xiangru Tang, Bolun Wang, Johan S. Wind, Stanislaw Wozniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Qinghua Zhou, Jian Zhu, Rui-Jie Zhu |
| Year | 2023 |
| arXiv | [2305.13048](https://arxiv.org/abs/2305.13048) |
| Status | full note started |

## Question

Transformers train efficiently because all token interactions can be processed in parallel, but dense attention has quadratic sequence-length cost:

$$
O(L^2d).
$$

RNNs decode efficiently because they maintain a compact recurrent state, but classic recurrent models are harder to train at scale and have struggled to match Transformer language-model quality.

RWKV asks:

$$
\text{Can one architecture train like a Transformer and infer like an RNN?}
$$

The proposed answer is a recurrent architecture with a linear-attention-like formulation that admits both views:

$$
\text{parallel training form}
\quad\leftrightarrow\quad
\text{recurrent inference form}.
$$

## Main Claim

RWKV's durable architecture claim is:

$$
\text{large-scale language modeling does not require dense self-attention at inference time.}
$$

The model combines:

| Component | Role |
| --- | --- |
| receptance gate | controls how much information is accepted |
| weighted key-value accumulation | linear-attention-like sequence memory |
| time mixing | mixes current token with previous-token information |
| channel mixing | token-wise feed-forward/channel update |

The paper reports scaling RWKV models up to 14B parameters and compares them with similarly sized Transformers. The key architectural point is not only the benchmark result; it is the dual formulation:

$$
\operatorname{RWKV}_{\text{train}}
\text{ can be parallelized,}
\qquad
\operatorname{RWKV}_{\text{infer}}
\text{ can be scanned recurrently.}
$$

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | causal token sequence |
| Target comparison | decoder-only Transformer language model |
| Sequence mixer | linear attention / recurrent weighted key-value state |
| Training mode | parallelizable over sequence |
| Inference mode | recurrent state update |
| Memory target | constant-size state per layer rather than KV cache growing with context |
| Natural task in paper | autoregressive language modeling |

The recurrent view is:

$$
s_t = F_\theta(s_{t-1}, x_t),
\qquad
y_t = G_\theta(s_t, x_t).
$$

The Transformer-like view computes the same kind of weighted accumulation over a whole sequence in parallel.

## Receptance Weighted Key Value

The name RWKV points to the core pieces:

$$
R,\ W,\ K,\ V.
$$

A simplified reading is:

| Symbol | Intuition |
| --- | --- |
| $R$ | receptance gate controlling output/read strength |
| $W$ | learned time decay or weighting over past keys/values |
| $K$ | key-like signal used in weighted accumulation |
| $V$ | value-like signal stored or mixed |

RWKV can be read as replacing pairwise self-attention:

$$
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}}
\right)V
$$

with a recurrently maintainable weighted key-value summary:

$$
s_t
=
\operatorname{Update}(s_{t-1}, k_t, v_t, w),
$$

$$
y_t
=
r_t \odot \operatorname{Read}(s_t).
$$

This is not the exact implementation formula, but it is the useful architecture contract: attention-like key/value accumulation without storing all previous keys and values.

## Time Mixing and Channel Mixing

RWKV keeps a block decomposition that resembles Transformer-style alternation between sequence mixing and channel mixing.

| Block | Transformer Analogue | RWKV Role |
| --- | --- | --- |
| time mixing | self-attention | recurrent/linear-attention-like sequence interaction |
| channel mixing | feed-forward network | token-wise feature transformation |

The pattern is:

$$
X
\xrightarrow{\text{time mixing}}
X'
\xrightarrow{\text{channel mixing}}
Y.
$$

This makes RWKV close to the architecture vocabulary used in [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]], [[papers/architectures/hyena|Hyena]], [[papers/architectures/s4|S4]], and [[papers/architectures/mamba|Mamba]]:

$$
\text{sequence mixer}
+
\text{channel mixer}
+
\text{residual/normalization scaffold}.
$$

## Inference State vs KV Cache

Decoder-only Transformers keep a KV cache whose memory grows with generated context length:

$$
\text{KV cache}
\propto
L \cdot n_{\mathrm{layers}}\cdot d.
$$

RWKV instead carries recurrent state:

$$
\text{state}
\propto
n_{\mathrm{layers}}\cdot d
$$

for fixed model width, independent of generated length. This is the practical reason RWKV matters for long-context or streaming generation.

| Architecture | Training Parallelism | Inference Memory |
| --- | --- | --- |
| Transformer | high | grows with KV cache length |
| classic RNN | limited over time | constant recurrent state |
| RWKV | parallelizable formulation | constant recurrent state |

## Relation to Other Sequence Alternatives

| Paper | Core Alternative to Dense Attention |
| --- | --- |
| [Transformer-XL](/papers/architectures/transformer-xl) | segment recurrence over Transformer hidden states |
| [Performer](/papers/architectures/performer) | kernelized linear attention approximation |
| [S4](/papers/architectures/s4) | structured state-space convolution/scan |
| [Hyena](/papers/architectures/hyena) | gated implicit long convolution |
| [Mamba](/papers/architectures/mamba) | selective state-space scan |
| [RWKV](/papers/architectures/rwkv) | Transformer-trainable recurrent language model |

RWKV is closest in spirit to the question:

$$
\text{Can recurrence return without giving up Transformer-era scaling?}
$$

## Evidence to Read Carefully

The paper reports large-scale language modeling experiments and pretrained models up to 14B parameters. Read the evidence through these checks:

| Evidence | What It Supports |
| --- | --- |
| scaling to 14B | recurrent-style models can be trained at modern LM scale |
| Transformer comparison | quality can be competitive at similar size |
| inference complexity | recurrent state avoids growing KV cache |
| training formulation | parallel version helps overcome classic RNN training bottleneck |

The key caveat is that "on par with Transformers" depends on data, tokenizer, optimizer, compute, benchmark, and model version. The architecture takeaway is stronger than any single leaderboard number.

## Limits

- RWKV removes dense attention, so explicit pairwise retrieval behavior may differ from Transformers.
- Constant recurrent state is a bottleneck: all past information must be compressed.
- Comparisons depend heavily on training recipe, data quality, and model scaling.
- The architecture has multiple versions; paper-level formulas may not match every released model variant.
- Some long-context tasks may need external retrieval or hybrid attention layers.

## What This Paper Teaches

RWKV forces a clean distinction:

$$
\text{Transformer-style training}
\neq
\text{attention-only inference}.
$$

When reading a new recurrent or attention-free language model, ask:

- Is the training form parallel over time?
- Is inference truly recurrent with constant state?
- What information is stored in the state?
- Does quality hold at comparable data and compute?
- Does the model handle retrieval-heavy tasks or only smooth language modeling?
- Is the comparison against optimized modern attention baselines?

## Concepts

- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/mamba|Mamba]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/llm/language-model|Language model]]

## Related

- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/long-short-term-memory|Long Short-Term Memory]]
- [[papers/architectures/transformer-xl|Transformer-XL]]
- [[papers/architectures/performer|Performer]]
- [[papers/architectures/s4|S4]]
- [[papers/architectures/hyena|Hyena]]
- [[papers/architectures/mamba|Mamba]]
