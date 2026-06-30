---
title: GLU Variants Improve Transformer
aliases:
  - papers/glu-variants-improve-transformer
  - papers/swiglu
  - papers/geglu
tags:
  - papers
  - architectures
  - transformer
  - feed-forward-network
  - gating
---

# GLU Variants Improve Transformer

> The paper showed that replacing the standard Transformer feed-forward activation with gated linear unit variants can improve model quality.

## Metadata

| Field | Value |
| --- | --- |
| Paper | GLU Variants Improve Transformer |
| Author | Noam Shazeer |
| Year | 2020 |
| Venue | arXiv preprint |
| arXiv | [2002.05202](https://arxiv.org/abs/2002.05202) |
| Related GLU paper | [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083) |
| Status | full note started |

## One-Line Takeaway

The paper turns the Transformer feed-forward network from a simple two-layer MLP into a gated token-wise block, making GLU-style FFNs a reusable architecture component for modern language models.

## Question

[[papers/architectures/attention-is-all-you-need|Attention Is All You Need]] alternates multi-head attention with a position-wise feed-forward network:

$$
\operatorname{FFN}(x)
=
W_2\phi(W_1x+b_1)+b_2.
$$

The usual reading focuses on attention, but a large fraction of Transformer parameters and compute is inside the FFN. The paper asks:

$$
\text{Can the FFN activation be replaced by a multiplicative gate?}
$$

More concretely:

$$
\text{ReLU/GELU MLP}
\quad\rightarrow\quad
\text{GLU-style gated MLP}.
$$

## Main Claim

The paper tests several GLU variants inside Transformer feed-forward sublayers and finds that some gated variants improve quality over standard ReLU or GELU FFNs.

The durable architecture claim is:

$$
\text{token-wise channel mixing}
+
\text{multiplicative feature gating}
\Rightarrow
\text{stronger Transformer FFN block}.
$$

This matters because later decoder-only models often use SwiGLU-style FFNs as part of their default block recipe, including [[papers/architectures/llama|LLaMA]]-style models.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | token representation $x\in\mathbb{R}^{d}$ |
| Output | token representation in $\mathbb{R}^{d}$ |
| Position mixing | none; applied independently per token |
| Channel mixing | yes, through linear projections and gating |
| Main operation | elementwise product between value and gate paths |
| Main replacement | FFN activation and first projection |
| Attention change | none |
| Common modern form | SwiGLU FFN |

For a token matrix:

$$
X\in\mathbb{R}^{T\times d},
$$

the FFN maps each row independently:

$$
Y_t = \operatorname{FFN}(X_t).
$$

This is why the block is "position-wise." It changes each token's feature vector after attention has already mixed information across tokens.

## Standard Transformer FFN

The original Transformer FFN can be written as:

$$
\operatorname{FFN}_{\text{ReLU}}(x)
=
\max(0, xW_1+b_1)W_2+b_2.
$$

Later Transformer variants often use GELU:

$$
\operatorname{FFN}_{\text{GELU}}(x)
=
\operatorname{GELU}(xW_1)W_2.
$$

The shape contract is usually:

$$
W_1\in\mathbb{R}^{d\times d_{\mathrm{ff}}},
\quad
W_2\in\mathbb{R}^{d_{\mathrm{ff}}\times d}.
$$

If $d_{\mathrm{ff}}=4d$, then the FFN has roughly:

$$
2d d_{\mathrm{ff}} = 8d^2
$$

projection parameters per block, ignoring biases. This is a major part of the model.

## GLU Form

The original GLU idea from gated convolutional language models uses two linear projections:

$$
\operatorname{GLU}(x)
=
\sigma(xW+b)
\odot
(xV+c).
$$

Here:

- $xW+b$ produces a gate pre-activation;
- $\sigma(\cdot)$ maps the gate into a soft on/off range;
- $xV+c$ produces the value stream;
- $\odot$ is elementwise multiplication.

The key difference from a normal activation is that the block has two learned paths:

$$
\text{gate path}
\quad\text{and}\quad
\text{value path}.
$$

The gate can suppress or pass features in a data-dependent way.

## GLU Variants

Shazeer tests variants that replace the sigmoid gate with other nonlinearities. A bias-free notation is:

$$
\operatorname{GLU}(x)
=
\sigma(xW)\odot xV.
$$

$$
\operatorname{Bilinear}(x)
=
(xW)\odot xV.
$$

$$
\operatorname{ReGLU}(x)
=
\operatorname{ReLU}(xW)\odot xV.
$$

$$
\operatorname{GEGLU}(x)
=
\operatorname{GELU}(xW)\odot xV.
$$

$$
\operatorname{SwiGLU}(x)
=
\operatorname{Swish}(xW)\odot xV.
$$

with:

$$
\operatorname{Swish}_{\beta}(z)
=
z\sigma(\beta z).
$$

The FFN form then applies an output projection:

$$
\operatorname{FFN}_{\text{SwiGLU}}(x)
=
\left[
\operatorname{Swish}(xW)
\odot
xV
\right]W_2.
$$

## Parameter-Matched Reading

A gated FFN has two input projections before the output projection:

$$
W,V\in\mathbb{R}^{d\times h},
\quad
W_2\in\mathbb{R}^{h\times d}.
$$

Approximate parameter count:

$$
2dh + hd = 3dh.
$$

A standard two-layer FFN with width $d_{\mathrm{ff}}$ has:

$$
2dd_{\mathrm{ff}}.
$$

To keep parameter count similar:

$$
3dh \approx 2dd_{\mathrm{ff}}
$$

so:

$$
h \approx \frac{2}{3}d_{\mathrm{ff}}.
$$

This is why many gated FFNs use a smaller hidden width than a standard $4d$ FFN. Without this adjustment, a gated FFN can look better simply because it has more parameters.

## Block View

Inside a pre-norm Transformer block:

$$
U_{\ell}
=
X_{\ell}
+
\operatorname{Attn}_{\ell}
\left(
\operatorname{Norm}(X_{\ell})
\right),
$$

$$
X_{\ell+1}
=
U_{\ell}
+
\operatorname{FFN}_{\text{GLU}}
\left(
\operatorname{Norm}(U_{\ell})
\right).
$$

The paper only changes the second residual branch:

$$
\operatorname{FFN}_{\text{ReLU/GELU}}
\quad\rightarrow\quad
\operatorname{FFN}_{\text{GLU variant}}.
$$

Attention, positional encoding, and the residual interface remain the same.

## Why Gating Helps

A normal activation transforms features independently:

$$
y_i = \phi(a_i).
$$

A gated FFN creates a multiplicative interaction:

$$
y_i = \phi(a_i)b_i.
$$

This lets one projected feature modulate another projected feature. In architecture terms:

| Mechanism | What It Adds |
| --- | --- |
| activation | nonlinear feature transform |
| gate | data-dependent feature selection |
| value stream | linear path that carries content |
| output projection | mixes gated hidden features back into model dimension |

The gate is local to a token's feature vector. It is not attention, routing, or memory by itself.

## Evidence to Read

The paper compares variants in Transformer sequence-to-sequence models.

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| GLU variants can beat ReLU/GELU | Transformer FFN ablation | FFN activation choice matters | model/task scale is narrower than modern LLM pretraining |
| GEGLU and SwiGLU are strong variants | variant comparison | smoother gated activations are useful | later adoption depends on recipe and parameter matching |
| FFN design is not a minor detail | replacing only FFN sublayer changes quality | token-wise MLP block is a core architecture component | gains can be confounded by hidden width and parameter budget |
| GLU transfers from convolutional LM ideas | link to original GLU work | multiplicative feature gating is reusable | original context was gated convolution, not Transformer FFN |

The main lesson is not "always use SwiGLU." It is that the FFN should be read as an architectural block with its own design space.

## Relation to Other Architecture Notes

| Paper | Similarity | Difference |
| --- | --- | --- |
| [Attention Is All You Need](/papers/architectures/attention-is-all-you-need) | defines the Transformer block with position-wise FFN | uses ReLU-style FFN rather than GLU variants |
| [LLaMA](/papers/architectures/llama) | uses SwiGLU-style gated FFNs in a modern decoder-only recipe | LLaMA combines SwiGLU with RMSNorm, RoPE, data, and scaling choices |
| [Root Mean Square Layer Normalization](/papers/architectures/root-mean-square-layer-normalization) | small repeated block in modern Transformers | RMSNorm normalizes scale; SwiGLU changes token-wise channel mixing |
| [Switch Transformer](/papers/architectures/switch-transformer) | changes FFN computation | Switch routes tokens to sparse experts; GLU variants modify the dense expert block |
| [T5](/papers/architectures/t5) | sequence-to-sequence Transformer context | T5 is a full text-to-text recipe, not a GLU-only paper |

## LLaMA-Style FFN

Many modern decoder-only models use a form close to:

$$
\operatorname{FFN}(x)
=
W_o
\left(
\operatorname{SiLU}(xW_g)
\odot
xW_u
\right).
$$

This is often described as SwiGLU, with naming conventions varying by implementation:

| Name in code | Role |
| --- | --- |
| `gate_proj` | produces gate pre-activation |
| `up_proj` | produces value stream |
| `down_proj` | projects gated hidden features back to $d$ |
| `intermediate_size` | gated hidden width |

When reading a model architecture table, "SwiGLU FFN" is not a decorative detail. It changes parameter count, activation statistics, and kernel shape.

## Limitations

- The original paper is short and focused on sequence-to-sequence Transformer experiments.
- It does not by itself prove SwiGLU is optimal for every model size, dataset, or modality.
- Gated FFNs can increase parameter count unless hidden width is adjusted.
- Runtime can depend on kernel fusion, memory bandwidth, and implementation layout.
- In modern LLMs, SwiGLU gains are entangled with [[papers/architectures/root-mean-square-layer-normalization|RMSNorm]], RoPE, data scale, optimizer settings, and training compute.

## Common Misreadings

| Misreading | Better Reading |
| --- | --- |
| SwiGLU is just an activation function | it is a gated FFN block with multiple projections |
| the paper changes attention | it changes the position-wise FFN sublayer |
| GLU variants are free | they change parameter count and compute unless width is adjusted |
| LLaMA is good because of SwiGLU alone | SwiGLU is one part of a broader recipe |
| FFNs are secondary to attention | FFNs hold a large part of Transformer parameters and compute |

## What to Remember

The core equation:

$$
\operatorname{FFN}_{\text{SwiGLU}}(x)
=
\left[
\operatorname{Swish}(xW)
\odot
xV
\right]W_2.
$$

The core reading rule:

$$
\text{Transformer block}
\neq
\text{attention only}.
$$

The FFN activation, gating structure, hidden width, and output projection are architecture decisions.

## Links

- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/architectures/transformer|Transformer]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/llama|LLaMA]]
- [[papers/architectures/root-mean-square-layer-normalization|Root Mean Square Layer Normalization]]
- [[papers/architectures/switch-transformer|Switch Transformer]]
