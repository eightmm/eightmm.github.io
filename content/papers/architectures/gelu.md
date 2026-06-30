---
title: Gaussian Error Linear Units
aliases:
  - papers/gelu
  - papers/gaussian-error-linear-units
tags:
  - papers
  - architectures
  - activation-function
  - transformer
---

# Gaussian Error Linear Units

> GELU replaces hard ReLU gating with a smooth probability-weighted activation, becoming a default nonlinearity in BERT-style and many later Transformer feed-forward blocks.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Gaussian Error Linear Units (GELUs) |
| Authors | Dan Hendrycks, Kevin Gimpel |
| Year | 2016 |
| Venue | arXiv preprint |
| arXiv | [1606.08415](https://arxiv.org/abs/1606.08415) |
| Code | [hendrycks/GELUs](https://github.com/hendrycks/GELUs) |
| Status | seed note started |

## One-Line Takeaway

GELU is the activation:

$$
\operatorname{GELU}(x)=x\Phi(x),
$$

where $\Phi$ is the standard Gaussian CDF, giving a smooth input-dependent gate rather than a hard threshold at zero.

## Question

ReLU applies a hard gate:

$$
\operatorname{ReLU}(x)
=
x\mathbf{1}_{x>0}.
$$

This is cheap and effective, but the gate is discontinuous in derivative at zero and fully suppresses all negative inputs.

GELU asks:

> Can an activation weight an input by how likely it is to be positive under a Gaussian uncertainty model?

## Formula

The GELU activation is:

$$
\operatorname{GELU}(x)
=
x\Phi(x),
$$

where:

$$
\Phi(x)
=
\frac{1}{2}
\left[
1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)
\right].
$$

So:

$$
\operatorname{GELU}(x)
=
\frac{x}{2}
\left[
1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)
\right].
$$

A common approximation is:

$$
\operatorname{GELU}(x)
\approx
0.5x
\left(
1+
\tanh
\left[
\sqrt{\frac{2}{\pi}}
(x+0.044715x^3)
\right]
\right).
$$

## Architecture Contract

| Component | Role |
| --- | --- |
| input $x$ | pre-activation value from linear layer |
| $\Phi(x)$ | smooth probability-like gate |
| product $x\Phi(x)$ | gates value by input magnitude and sign |
| approximation | faster implementation option |
| feed-forward placement | activation inside MLP/FFN block |

In a Transformer feed-forward block:

$$
\operatorname{FFN}(x)
=
W_2\,\sigma(W_1x+b_1)+b_2,
$$

GELU sets:

$$
\sigma=\operatorname{GELU}.
$$

This is why GELU matters as an architecture component: it changes the nonlinearity in the token-wise channel mixer.

## ReLU vs GELU

| Axis | ReLU | GELU |
| --- | --- | --- |
| Formula | $\max(0,x)$ | $x\Phi(x)$ |
| Gate | hard sign gate | smooth probability-weighted gate |
| Negative values | all zeroed | small negatives can pass |
| Derivative | kink at zero | smooth |
| Common use | CNNs and many MLPs | BERT-style Transformers and modern FFNs |

GELU is not a new macro-architecture. It is a small block-level choice that became durable because activation behavior compounds over many layers.

## Derivative

Let $\phi(x)$ be the standard Gaussian density:

$$
\phi(x)
=
\frac{1}{\sqrt{2\pi}}\exp(-x^2/2).
$$

Then:

$$
\frac{d}{dx}\operatorname{GELU}(x)
=
\Phi(x)+x\phi(x).
$$

This derivative is smooth. For large positive $x$:

$$
\Phi(x)\approx 1
\Rightarrow
\operatorname{GELU}(x)\approx x.
$$

For large negative $x$:

$$
\Phi(x)\approx 0
\Rightarrow
\operatorname{GELU}(x)\approx 0.
$$

Near zero, GELU behaves like a soft gate instead of an abrupt switch.

## Why It Matters

GELU became important because many Transformer blocks are built from:

$$
\text{attention}
+
\text{token-wise FFN}
+
\text{normalization}
+
\text{residual paths}.
$$

The FFN is often where much of the parameter count lives:

$$
d_{\text{model}}
\rightarrow
d_{\text{ff}}
\rightarrow
d_{\text{model}}.
$$

So the activation inside the FFN is not cosmetic. It affects channel mixing, gradient flow, sparsity, saturation, and training stability.

## What To Watch

- GELU is usually an activation choice, not a standalone architecture claim.
- Exact and approximate GELU can differ slightly; implementations should be recorded.
- Performance gains are recipe-dependent and can interact with normalization, initialization, optimizer, and depth.
- In modern LLMs, GELU is often replaced by gated activations such as SwiGLU; compare with [[papers/architectures/glu-variants-improve-transformer|GLU Variants Improve Transformer]].
- If a paper claims an architecture improvement, isolate whether the gain came from activation, width, normalization, or training recipe.

## Related

- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/gating|Gating]]
- [[papers/architectures/bert|BERT]]
- [[papers/architectures/glu-variants-improve-transformer|GLU Variants Improve Transformer]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
