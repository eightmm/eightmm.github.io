---
title: Root Mean Square Layer Normalization
aliases:
  - papers/rmsnorm
  - papers/root-mean-square-layer-normalization
tags:
  - papers
  - architectures
  - normalization
  - transformer
---

# Root Mean Square Layer Normalization

> The paper introduced RMSNorm: a lighter LayerNorm variant that keeps scale normalization but removes mean-centering.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Root Mean Square Layer Normalization |
| Authors | Biao Zhang, Rico Sennrich |
| Year | 2019 |
| Venue | NeurIPS 2019 |
| arXiv | [1910.07467](https://arxiv.org/abs/1910.07467) |
| Proceedings | [NeurIPS abstract](https://proceedings.neurips.cc/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html) |
| Code | [bzhangGo/rmsnorm](https://github.com/bzhangGo/rmsnorm) |
| Status | full note started |

## One-Line Takeaway

RMSNorm keeps the most useful part of [[papers/architectures/layer-normalization|Layer Normalization]] for many sequence models: rescaling a hidden vector by its feature-wise magnitude, while skipping mean subtraction and the usual additive shift.

## Question

LayerNorm normalizes a hidden vector by both centering and scaling:

$$
\operatorname{LayerNorm}(x)
=
\gamma
\odot
\frac{x-\mu(x)}{\sqrt{\sigma^2(x)+\epsilon}}
+
\beta.
$$

RMSNorm asks:

$$
\text{Do we need both recentering invariance and rescaling invariance?}
$$

The paper's hypothesis is that, for many neural architectures, the rescaling part is the more important stabilizing property. If that is true, the normalization block can be simpler:

$$
\text{remove mean subtraction}
\quad\Rightarrow\quad
\text{less computation, similar training behavior}.
$$

## Main Claim

RMSNorm normalizes by the root mean square of the hidden vector:

$$
\operatorname{RMS}(x)
=
\sqrt{
\frac{1}{d}
\sum_{i=1}^{d}
x_i^2
}.
$$

Then:

$$
\operatorname{RMSNorm}(x)
=
g
\odot
\frac{x}{\operatorname{RMS}(x)+\epsilon}.
$$

The durable architecture claim is:

$$
\text{feature-wise RMS scaling}
\approx
\text{LayerNorm-like stabilization}
\quad
\text{with a smaller normalization contract}.
$$

This does not mean RMSNorm is always better than LayerNorm. It means the normalization family should be read by its axis and invariance contract, not only by its name.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | hidden vector or token residual-stream vector $x\in\mathbb{R}^{d}$ |
| Output | same shape as input |
| Statistic | root mean square across feature dimension |
| Mean subtraction | no |
| Learned scale | usually yes, $g\in\mathbb{R}^{d}$ |
| Learned shift | usually no in the canonical RMSNorm form |
| Train/eval behavior | same formula; no running statistics |
| Main invariant | rescaling invariance |
| Common placement | pre-norm Transformer blocks and recurrent blocks |

For a sequence tensor:

$$
X\in\mathbb{R}^{B\times T\times d},
$$

RMSNorm is usually applied independently per batch item and token:

$$
r_{bt}
=
\sqrt{
\frac{1}{d}
\sum_{j=1}^{d}
X_{btj}^{2}
}.
$$

$$
Y_{btj}
=
g_j
\frac{X_{btj}}{r_{bt}+\epsilon}.
$$

Batch index $b$ and token index $t$ are not pooled. Only the feature axis $d$ is normalized.

## LayerNorm vs RMSNorm

The difference is not cosmetic. The two blocks normalize different statistics:

| Dimension | LayerNorm | RMSNorm |
| --- | --- | --- |
| Centering | subtracts $\mu(x)$ | does not subtract mean |
| Scale statistic | standard deviation after centering | root mean square |
| Additive bias | usually $\beta$ | usually omitted |
| Learned scale | $\gamma$ | $g$ |
| Invariance emphasis | recentering and rescaling | rescaling |
| Running stats | no | no |
| Common modern use | general sequence models | decoder-only LLM blocks |

For LayerNorm:

$$
\mu(x)
=
\frac{1}{d}
\sum_{i=1}^{d}
x_i
$$

$$
\sigma^2(x)
=
\frac{1}{d}
\sum_{i=1}^{d}
(x_i-\mu(x))^2.
$$

For RMSNorm:

$$
\operatorname{RMS}^2(x)
=
\frac{1}{d}
\sum_{i=1}^{d}
x_i^2.
$$

If $x$ has zero mean, the two scale terms are closely related:

$$
\mu(x)=0
\quad\Rightarrow\quad
\operatorname{RMS}^2(x)=\sigma^2(x).
$$

If $x$ has a large mean offset, they differ:

$$
\operatorname{RMS}^2(x)
=
\sigma^2(x)+\mu(x)^2.
$$

This identity explains the architectural tradeoff. RMSNorm keeps the mean component in the vector and normalizes total magnitude. LayerNorm removes the mean component first.

## Invariance Reading

If the input is scaled by a positive scalar $a$:

$$
x' = a x,
$$

then:

$$
\operatorname{RMS}(x')
=
a\operatorname{RMS}(x).
$$

So, ignoring $\epsilon$:

$$
\frac{x'}{\operatorname{RMS}(x')}
=
\frac{a x}{a\operatorname{RMS}(x)}
=
\frac{x}{\operatorname{RMS}(x)}.
$$

This gives RMSNorm its rescaling invariance:

$$
\operatorname{RMSNorm}(a x)
\approx
\operatorname{RMSNorm}(x).
$$

But RMSNorm is not recentering invariant. If:

$$
x' = x + c\mathbf{1},
$$

then the RMS changes and the mean offset remains part of the normalized representation. The paper's bet is that removing this recentering step is often acceptable.

## Partial RMSNorm

The paper also proposes partial RMSNorm, or pRMSNorm. Instead of estimating RMS from every feature:

$$
\operatorname{RMS}(x)
=
\sqrt{
\frac{1}{d}
\sum_{i=1}^{d}
x_i^2
},
$$

pRMSNorm estimates it from a subset $S$ of features:

$$
\operatorname{pRMS}(x;S)
=
\sqrt{
\frac{1}{|S|}
\sum_{i\in S}
x_i^2
}.
$$

The idea is:

$$
|S|<d
\quad\Rightarrow\quad
\text{less normalization compute}
$$

while preserving the same style of rescaling invariance in expectation. For most modern LLM reading, the standard full-feature RMSNorm is the important artifact; pRMSNorm is useful as a reminder that normalization statistics are also an efficiency design choice.

## Block View

RMSNorm is usually not read as a standalone architecture family. It is a block inside a residual architecture.

Pre-norm Transformer form:

$$
U_{\ell}
=
X_{\ell}
+
\operatorname{Attn}_{\ell}
\left(
\operatorname{RMSNorm}(X_{\ell})
\right),
$$

$$
X_{\ell+1}
=
U_{\ell}
+
\operatorname{FFN}_{\ell}
\left(
\operatorname{RMSNorm}(U_{\ell})
\right).
$$

This is why RMSNorm appears in [[papers/architectures/llama|LLaMA]]-style models together with:

| Component | Role |
| --- | --- |
| RMSNorm | residual-stream scale control |
| RoPE | relative position behavior in attention |
| SwiGLU | gated feed-forward nonlinearity |
| pre-norm layout | stable residual path |
| causal self-attention | autoregressive token modeling |

The individual block is small, but the combination became a standard decoder-only Transformer recipe.

## Evidence to Read

The paper evaluates RMSNorm against LayerNorm across sequence-model settings and reports similar performance with lower runtime overhead.

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| recentering can be removed | LayerNorm vs RMSNorm comparison | rescaling may be enough in tested models | task/model scale predates modern trillion-token LLMs |
| RMSNorm is cheaper | operation count and runtime | removing mean subtraction reduces overhead | real gains depend on kernel fusion and hardware |
| pRMSNorm can approximate full RMSNorm | subset statistic experiments | statistic estimation is a design axis | less common in current foundation model recipes |
| sequence models tolerate RMSNorm | RNN/Transformer-style experiments | normalization contract transfers across architectures | later LLM adoption also depends on broader recipe |

Do not read the paper as "RMSNorm always beats LayerNorm." Read it as a block-level simplification with a clear invariance claim.

## Relation to Other Architecture Notes

| Paper | Similarity | Difference |
| --- | --- | --- |
| [Layer Normalization](/papers/architectures/layer-normalization) | per-example feature-axis normalization | LayerNorm centers and scales; RMSNorm scales by RMS only |
| [Attention Is All You Need](/papers/architectures/attention-is-all-you-need) | Transformer blocks need normalization for stability | original Transformer used LayerNorm rather than RMSNorm |
| [LLaMA](/papers/architectures/llama) | modern decoder-only stack uses RMSNorm | LLaMA is a full model recipe, not only a normalization paper |
| [Batch Normalization](/papers/architectures/batch-normalization) | normalizes activations to stabilize training | BatchNorm uses batch statistics and usually train/eval modes |
| [FlashAttention](/papers/architectures/flashattention) | efficiency matters for common Transformer primitives | FlashAttention changes attention scheduling; RMSNorm changes normalization contract |

## Why This Matters

RMSNorm is easy to overlook because it is a small equation. But small blocks matter when they are repeated in every layer and every token.

For a deep decoder-only Transformer:

$$
\text{norm calls}
\approx
2L
\quad
\text{per forward pass},
$$

where $L$ is the number of layers. During training, each call also participates in backward computation. At large scale:

$$
\text{small per-token block cost}
\times
\text{layers}
\times
\text{tokens}
\Rightarrow
\text{meaningful training and inference cost}.
$$

RMSNorm also matters for reading modern model cards and architecture tables. If a paper says only "Transformer backbone," you still need to ask:

- Is the block pre-norm or post-norm?
- Is the normalizer LayerNorm or RMSNorm?
- Does the normalizer include bias?
- Is the final output norm separate?
- Are normalization parameters exempted from weight decay?

## Limitations

- RMSNorm does not remove mean offsets, so it is not a drop-in conceptual replacement when centering is important.
- Runtime gains depend on implementation quality; unfused kernels may hide or exaggerate the benefit.
- Paper-scale experiments are much smaller than modern foundation model training.
- pRMSNorm adds estimator choices that can complicate reproducibility.
- In a full model, gains from RMSNorm are entangled with residual layout, optimizer, initialization, learning-rate schedule, data, and compute.

## Common Misreadings

| Misreading | Better Reading |
| --- | --- |
| RMSNorm is just LayerNorm with a different name | it removes mean subtraction and usually the additive shift |
| RMSNorm explains LLaMA performance | it is one part of the LLaMA-style block recipe |
| faster normalization always means faster model | attention, MLP, memory bandwidth, and kernel fusion may dominate |
| RMSNorm is always better | it trades recentering invariance for a simpler scale contract |
| normalization is only an optimization detail | repeated norm placement is part of architecture design |

## What to Remember

RMSNorm's core equation is:

$$
y
=
g
\odot
\frac{x}{
\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2}
+
\epsilon
}.
$$

The core conceptual move is:

$$
\text{LayerNorm}
-
\text{mean-centering}
\Rightarrow
\text{RMSNorm}.
$$

For architecture reading, RMSNorm should be tracked as a first-class block whenever a paper discusses modern decoder-only Transformers, long-context models, or efficient sequence modeling.

## Links

- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/normalization-placement|Normalization placement]]
- [[papers/architectures/layer-normalization|Layer Normalization]]
- [[papers/architectures/batch-normalization|Batch Normalization]]
- [[papers/architectures/llama|LLaMA]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
