---
title: GQA
aliases:
  - papers/gqa
  - papers/grouped-query-attention
  - papers/multi-query-attention
  - papers/fast-transformer-decoding
tags:
  - papers
  - architectures
  - attention
  - transformer
  - inference
---

# GQA

> Grouped-query attention reduces key-value cache cost by letting multiple query heads share fewer key/value heads.

## Metadata

| Field | Value |
| --- | --- |
| Paper | GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints |
| Authors | Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, Sumit Sanghai |
| Year | 2023 |
| Venue | EMNLP 2023 |
| arXiv | [2305.13245](https://arxiv.org/abs/2305.13245) |
| ACL Anthology | [2023.emnlp-main.298](https://aclanthology.org/2023.emnlp-main.298/) |
| Precursor | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) |
| Status | full note started |

## One-Line Takeaway

GQA interpolates between full multi-head attention and multi-query attention: keep many query heads for expressivity, but use fewer key/value heads to reduce autoregressive decoding memory bandwidth.

## Question

[[papers/architectures/attention-is-all-you-need|Attention Is All You Need]] uses multi-head attention, where each head has its own query, key, and value projections:

$$
Q_h=XW_Q^{(h)},\quad
K_h=XW_K^{(h)},\quad
V_h=XW_V^{(h)}.
$$

This is expressive, but autoregressive decoding stores and reloads key/value cache tensors for every layer and generated token. The paper asks:

$$
\text{Can we reduce KV cache cost without losing most multi-head quality?}
$$

The route is:

$$
\text{many query heads}
\quad+\quad
\text{fewer key/value heads}.
$$

## Main Claim

Grouped-query attention divides query heads into groups. Query heads inside a group share one key head and one value head.

The main architecture claim is:

$$
\text{GQA}
\approx
\text{MHA quality}
\quad\text{with}\quad
\text{MQA-like decoding speed}.
$$

The paper also proposes uptraining: convert a multi-head checkpoint into an MQA/GQA checkpoint, then continue pretraining for a small fraction of the original compute.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | decoder hidden states |
| Output | same attention output shape as MHA |
| Query heads | many, $H_q$ |
| Key/value heads | fewer, $H_{kv}$ |
| Sharing rule | several query heads share one key/value head |
| Main benefit | smaller KV cache and lower memory bandwidth during decoding |
| Main risk | quality loss if too few KV heads |
| Related variants | MHA, MQA, GQA |

In a decoder-only model, cached keys and values dominate attention memory during long generation:

$$
\text{KV cache size}
\propto
L\cdot T\cdot H_{kv}\cdot d_h,
$$

where:

- $L$ is number of layers;
- $T$ is cached sequence length;
- $H_{kv}$ is number of key/value heads;
- $d_h$ is head dimension.

Reducing $H_{kv}$ directly reduces KV cache size and memory bandwidth.

## MHA, MQA, GQA

Let $H_q$ be the number of query heads and $H_{kv}$ be the number of key/value heads.

| Attention type | Query heads | Key/value heads | Reading |
| --- | --- | --- | --- |
| MHA | $H_q=H$ | $H_{kv}=H$ | each query head has its own key/value head |
| MQA | $H_q=H$ | $H_{kv}=1$ | all query heads share one key/value head |
| GQA | $H_q=H$ | $1 < H_{kv} < H$ | groups of query heads share key/value heads |

MHA:

$$
K_h=XW_K^{(h)},\quad V_h=XW_V^{(h)}
\quad\text{for each }h.
$$

MQA:

$$
K=XW_K,\quad V=XW_V
\quad\text{shared by all query heads}.
$$

GQA:

$$
K_g=XW_K^{(g)},\quad V_g=XW_V^{(g)}
\quad\text{for group }g.
$$

For query head $h$, define its group:

$$
g(h)=\left\lfloor \frac{hH_{kv}}{H_q}\right\rfloor.
$$

Then:

$$
\operatorname{head}_h
=
\operatorname{Attention}
\left(
Q_h,\,
K_{g(h)},\,
V_{g(h)}
\right).
$$

The output concatenation contract stays the same:

$$
\operatorname{GQA}(X)
=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_{H_q})W_O.
$$

## Why Decoding Is Different From Training

During full-sequence training, attention can be computed in parallel over tokens. During autoregressive generation, tokens are produced one at a time:

$$
x_1,\ldots,x_t
\rightarrow
x_{t+1}.
$$

At each step, the model reuses cached keys and values from earlier tokens:

$$
K_{\leq t},V_{\leq t}.
$$

The bottleneck is often not only floating-point compute. It is reading the KV cache from memory:

$$
\text{decode latency}
\approx
\text{weight reads}
+
\text{KV cache reads}
+
\text{attention/MLP compute}.
$$

MQA and GQA target the KV cache term. This is why they are architecture changes with systems consequences.

## Uptraining

The GQA paper also gives a checkpoint conversion recipe. Starting from an MHA checkpoint, key/value projections can be pooled or grouped, then the model is further trained:

$$
\theta_{\text{MHA}}
\rightarrow
\theta_{\text{GQA init}}
\rightarrow
\theta_{\text{GQA uptrained}}.
$$

The stated motivation is practical:

$$
\text{reuse a strong MHA checkpoint}
\quad+\quad
\text{spend limited extra pretraining compute}.
$$

This matters because training a new large model only to change head sharing can be expensive.

## Relation to Multi-Query Attention

The precursor paper, "Fast Transformer Decoding: One Write-Head is All You Need," introduced multi-query attention:

$$
H_{kv}=1.
$$

MQA gives the strongest KV-cache reduction, but can hurt quality. GQA is the intermediate point:

$$
1 < H_{kv} < H_q.
$$

This makes GQA a more flexible design axis:

$$
\text{quality}
\leftrightarrow
\text{KV-cache memory}
\leftrightarrow
\text{decode throughput}.
$$

## Evidence to Read

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| MQA speeds decoding | MQA precursor experiments | sharing KV heads reduces memory bandwidth | quality can degrade |
| GQA preserves more quality | GQA comparisons against MHA/MQA | intermediate KV sharing is a useful tradeoff | results depend on model size and uptraining recipe |
| uptraining is feasible | checkpoint conversion plus extra pretraining | existing checkpoints can be adapted | still requires nontrivial compute and original recipe access |
| KV cache is a decoding bottleneck | performance analysis | attention architecture affects serving cost | hardware and implementation details matter |

The important reading is not just "GQA is faster." It is:

$$
\text{head topology}
\Rightarrow
\text{KV cache shape}
\Rightarrow
\text{serving behavior}.
$$

## Relation to Other Architecture Notes

| Paper | Similarity | Difference |
| --- | --- | --- |
| [Attention Is All You Need](/papers/architectures/attention-is-all-you-need) | defines multi-head attention | GQA changes key/value head sharing |
| [FlashAttention](/papers/architectures/flashattention) | improves attention efficiency | FlashAttention changes kernel scheduling; GQA changes attention head topology |
| [LLaMA](/papers/architectures/llama) | later LLaMA-family models often care about GQA/MQA choices | LLaMA is a full model recipe, not a head-sharing paper |
| [Transformer-XL](/papers/architectures/transformer-xl) | improves long-context use | Transformer-XL adds segment recurrence; GQA reduces KV cache cost |
| [RoFormer](/papers/architectures/roformer) | modifies attention internals | RoPE changes position handling; GQA changes KV sharing |

## Implementation Reading

In model config, watch for names like:

| Config Field | Meaning |
| --- | --- |
| `num_attention_heads` | number of query heads $H_q$ |
| `num_key_value_heads` | number of key/value heads $H_{kv}$ |
| `n_head` | often query heads |
| `n_kv_head` | often key/value heads |
| `head_dim` | per-head feature dimension |

If:

$$
H_q=32,\quad H_{kv}=8,
$$

then each key/value head serves:

$$
\frac{H_q}{H_{kv}}=4
$$

query heads.

The implementation must repeat or broadcast key/value heads to query-head groups before attention score computation. Bugs often come from confusing:

- query-head count;
- key/value-head count;
- tensor layout order;
- cache layout;
- grouped broadcast semantics.

## Limitations

- GQA is not a new reasoning mechanism; it is a head-sharing and serving-efficiency design.
- If $H_{kv}$ is too small, quality may degrade.
- Uptraining assumes access to the checkpoint and enough data/compute to adapt.
- Speedup depends on KV cache layout, batch size, sequence length, hardware, and kernel implementation.
- Comparisons must separate architecture effect from model size, data, tokenizer, and serving stack.

## Common Misreadings

| Misreading | Better Reading |
| --- | --- |
| GQA changes the attention formula completely | it keeps dot-product attention but shares K/V heads |
| GQA is the same as FlashAttention | GQA changes tensors; FlashAttention changes computation schedule |
| MQA/GQA only matter for inference engineers | the architecture determines KV cache shape and deployability |
| fewer KV heads always means better | it trades memory bandwidth against representation capacity |
| uptraining is free | it is cheaper than full pretraining, but still a training procedure |

## What to Remember

The core axis is:

$$
H_{kv}
\in
\{1,\ldots,H_q\}.
$$

With:

$$
\text{MQA}: H_{kv}=1,
\quad
\text{GQA}: 1<H_{kv}<H_q,
\quad
\text{MHA}: H_{kv}=H_q.
$$

The reason it matters:

$$
\text{fewer KV heads}
\Rightarrow
\text{smaller KV cache}
\Rightarrow
\text{faster/cheaper autoregressive decoding}.
$$

## Links

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/flashattention|FlashAttention]]
- [[papers/architectures/llama|LLaMA]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
