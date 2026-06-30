---
title: Jamba
aliases:
  - papers/jamba
  - papers/hybrid-transformer-mamba
  - papers/jamba-hybrid-transformer-mamba-language-model
tags:
  - papers
  - architectures
  - transformer
  - mamba
  - state-space-model
  - mixture-of-experts
  - language-models
---

# Jamba

> Jamba is useful because it treats Transformer attention, Mamba-style state-space layers, and sparse MoE routing as composable architecture choices rather than mutually exclusive model families.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Jamba: A Hybrid Transformer-Mamba Language Model |
| Authors | Opher Lieber, Barak Lenz, Hofit Bata, Gal Cohen, Jhonathan Osin, Itay Dalmedigos, Erez Safahi, Shaked Meirom, Yonatan Belinkov, Shai Shalev-Shwartz, Omri Abend, Raz Alon, Tomer Asida, Amir Bergman, Roman Glozman, Michael Gokhman, Avashalom Manevich, Nir Ratner, Noam Rozen, Erez Shwartz, Mor Zusman, Yoav Shoham |
| Year | 2024 |
| Venue | arXiv preprint |
| arXiv | [2403.19887](https://arxiv.org/abs/2403.19887) |
| Model | [ai21labs/Jamba-v0.1](https://huggingface.co/ai21labs/Jamba-v0.1) |
| Status | full note started |

## Question

Decoder-only Transformers are strong language-model backbones, but long contexts expose a memory problem:

$$
\text{KV cache}
\propto
L \cdot T \cdot d,
$$

where $L$ is the number of attention layers, $T$ is context length, and $d$ is hidden size.

Mamba-style state-space models reduce the need to store every previous key/value pair:

$$
h_t
=
A_t h_{t-1}
+
B_t x_t,
\qquad
y_t=C_t h_t.
$$

But a pure SSM can lose some of the explicit token-token retrieval behavior that attention provides.

Jamba asks:

$$
\text{Can a large language model combine attention retrieval, Mamba efficiency, and MoE capacity in one backbone?}
$$

## Main Claim

The durable architecture claim is:

$$
\text{Transformer layers}
+
\text{Mamba layers}
+
\text{sparse MoE FFN layers}
\Rightarrow
\text{long-context LM with lower KV-cache pressure and high active capacity}.
$$

This makes Jamba a hybrid architecture note, not only a model release note. Its main contribution is the composition rule:

| Component | Role |
| --- | --- |
| Transformer attention | explicit content-based token-token mixing |
| Mamba layer | efficient recurrent/state-space sequence mixing |
| MoE feed-forward layer | larger total parameter capacity with sparse active compute |

For this wiki, Jamba belongs between [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]], [[papers/architectures/mamba|Mamba]], [[papers/architectures/mamba-2|Mamba-2]], and [[papers/architectures/glam|GLaM]].

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | causal token sequence |
| Output | next-token distribution |
| Base family | decoder-only language model |
| Sequence mixers | interleaved attention and Mamba layers |
| Capacity mechanism | sparse MoE in selected feed-forward layers |
| Main bottleneck addressed | KV-cache memory and long-context throughput |
| Reported implementation | 12B active parameters, 52B total parameters |
| Reported context target | up to 256K tokens |

The training objective remains the ordinary autoregressive loss:

$$
\mathcal{L}
=
-\sum_{t=1}^{T}
\log p_\theta(x_t \mid x_{<t}).
$$

The novelty is not a new loss. It is the backbone allocation of attention, SSM, and sparse experts.

## Hybrid Block View

A simplified Jamba layer stack can be read as:

$$
H^{\ell+1}
=
\operatorname{Block}_{\ell}(H^\ell),
$$

where each block chooses a sequence mixer:

$$
\operatorname{Mixer}_{\ell}
\in
\{
\operatorname{Attention},
\operatorname{Mamba}
\}.
$$

After the mixer, a feed-forward path may be dense or sparse:

$$
\operatorname{FFN}_{\ell}
\in
\{
\operatorname{DenseFFN},
\operatorname{MoE\text{-}FFN}
\}.
$$

So an abstract layer is:

$$
\tilde{H}^{\ell}
=
H^\ell
+
\operatorname{Mixer}_{\ell}(\operatorname{Norm}(H^\ell)),
$$

$$
H^{\ell+1}
=
\tilde{H}^{\ell}
+
\operatorname{FFN}_{\ell}(\operatorname{Norm}(\tilde{H}^{\ell})).
$$

This expression is intentionally generic. The paper's key degree of freedom is how often to use attention, how often to use Mamba, and how often to replace dense FFNs with MoE.

## KV Cache Pressure

Attention layers need a KV cache during autoregressive decoding:

$$
K_{1:t}, V_{1:t}
\in
\mathbb{R}^{t \times d}.
$$

If every layer is attention, cache memory grows with both context length and number of layers:

$$
M_{\mathrm{KV}}
=
O(L_{\mathrm{attn}} T d).
$$

Hybridizing with Mamba reduces the number of attention layers that need full KV cache:

$$
L_{\mathrm{attn}}
<
L_{\mathrm{total}}.
$$

The paper reports a much smaller KV-cache footprint for Jamba than comparable attention-heavy open models at very long context. The architecture lesson is:

$$
\text{long-context memory}
\text{ is controlled by which layers require explicit past-token storage}.
$$

## MoE Path

Jamba also inserts sparse MoE in some feed-forward paths. For a token representation $x_t$, a router computes:

$$
r_t
=
\operatorname{softmax}(W_r x_t),
$$

then selects top experts:

$$
S_t
=
\operatorname{TopK}(r_t,k).
$$

The sparse output is:

$$
\operatorname{MoE}(x_t)
=
\sum_{i\in S_t}
\alpha_{t,i}E_i(x_t).
$$

This separates total model capacity from active compute:

$$
|\theta|_{\mathrm{total}}
>
|\theta|_{\mathrm{active}}(x_t).
$$

For Jamba, this is important because Mamba reduces long-context memory pressure while MoE increases available model capacity without activating every parameter for every token.

## Why This Paper Belongs in Architecture

Jamba is not primarily a new benchmark, dataset, or training objective paper. It is an architecture composition paper.

| Axis | Jamba's Contribution |
| --- | --- |
| sequence mixing | interleaves attention and Mamba layers |
| memory | reduces KV cache by not making every layer attention-based |
| capacity | uses MoE to increase total parameters with sparse activation |
| scaling question | tests the hybrid at large language-model scale |
| ablation target | attention/Mamba ratio, MoE placement, explicit positional information, Mamba stability |

This makes it a good example of how paper routing should work in this blog:

- it lives in [Architecture papers](/papers/architectures);
- it links to [[concepts/architectures/state-space-model|state-space models]] and [[concepts/architectures/mixture-of-experts|Mixture of Experts]];
- it can still be mentioned from systems or long-context notes when the focus is serving memory.

## Relation to Mamba and Mamba-2

[[papers/architectures/mamba|Mamba]] asks whether selective state-space layers can compete with attention as sequence mixers.

[[papers/architectures/mamba-2|Mamba-2]] asks whether SSMs and attention can be understood through a shared structured matrix view.

Jamba asks a different practical question:

$$
\text{If attention and Mamba have complementary strengths, what happens when they are interleaved?}
$$

| Paper | Architecture Lesson |
| --- | --- |
| [Mamba](/papers/architectures/mamba) | selective SSM as an efficient sequence backbone |
| [Mamba-2](/papers/architectures/mamba-2) | SSM/attention duality through structured matrix mixers |
| [Jamba](/papers/architectures/jamba) | hybrid attention-Mamba-MoE allocation at large LM scale |

The distinction matters. Jamba is not evidence that pure Mamba replaces attention; it is evidence that mixed backbones are a serious design point.

## Relation to MoE Scaling

Jamba inherits the sparse-capacity idea from MoE language models such as [[papers/architectures/gshard|GShard]], [[papers/architectures/switch-transformer|Switch Transformer]], and [[papers/architectures/glam|GLaM]].

The difference is the base sequence mixer:

| Model | Sequence Backbone | Sparse Capacity |
| --- | --- | --- |
| [Switch Transformer](/papers/architectures/switch-transformer) | Transformer | top-1 MoE FFN |
| [GLaM](/papers/architectures/glam) | decoder-only Transformer | top-2 MoE FFN |
| Jamba | Transformer + Mamba hybrid | MoE FFN in selected layers |

This makes Jamba a useful bridge between conditional compute and state-space sequence modeling.

## Evidence to Read Carefully

Jamba reports academic benchmark performance, long-context evaluations, throughput analysis, KV-cache comparison, and ablations.

The main evidence claims should be separated:

| Evidence | Supports | Does Not Prove Alone |
| --- | --- | --- |
| benchmark scores | hybrid backbone can be competitive | universal superiority over dense Transformers |
| long-context evaluations | long-context behavior under chosen tasks | robust retrieval for every long-context workload |
| KV-cache comparison | attention-layer count matters for memory | total serving cost without batch/hardware details |
| throughput measurement | implementation can be faster in reported setting | architecture-only gain independent of kernel stack |
| MoE ablation | sparse capacity helps the hybrid | expert specialization or easy deployment |
| Mamba RMSNorm stabilization | large-scale Mamba layers can need internal stabilization | all SSM variants are unstable |

The paper is most useful when read as a design-space map rather than a final answer.

## Practical Checks

- Count attention layers separately from total layers.
- Track active parameters and total parameters separately.
- Check whether MoE routing uses top-1, top-2, capacity limits, or dropped tokens.
- Compare KV-cache memory at the same context length, dtype, hidden size, and layer count.
- Separate long-context retrieval quality from ordinary benchmark quality.
- Check whether positional information is explicit, implicit, or absent in each layer type.
- Treat throughput numbers as hardware- and implementation-dependent.

## Where It Fits

Jamba is a strong note for the architecture shelf because it combines three major routes:

$$
\text{attention}
\quad+\quad
\text{state-space sequence modeling}
\quad+\quad
\text{sparse expert capacity}.
$$

For future notes, use Jamba as the bridge when reading:

- hybrid Transformer-SSM language models;
- long-context backbones with reduced KV cache;
- sparse expert models where the base layer is not purely Transformer;
- model-family comparisons that mix architecture, systems, and scaling claims.

## Related

- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/mamba|Mamba]]
- [[concepts/architectures/mixture-of-experts|Mixture of Experts]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/mamba|Mamba]]
- [[papers/architectures/mamba-2|Mamba-2]]
- [[papers/architectures/s4|S4]]
- [[papers/architectures/hyena|Hyena]]
- [[papers/architectures/retnet|RetNet]]
- [[papers/architectures/switch-transformer|Switch Transformer]]
- [[papers/architectures/glam|GLaM]]
