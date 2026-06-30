---
title: Feed-Forward Network
tags:
  - architectures
  - neural-networks
---

# Feed-Forward Network

A feed-forward network maps each position independently through dense layers and nonlinearities. In Transformers, the feed-forward network is the token-wise MLP between attention blocks.

For token matrix $X\in\mathbb{R}^{T\times d}$:

$$
H = \sigma(XW_1 + b_1)
$$

$$
Y = HW_2 + b_2
$$

where $W_1\in\mathbb{R}^{d\times d_{\mathrm{ff}}}$, $W_2\in\mathbb{R}^{d_{\mathrm{ff}}\times d}$, $d_{\mathrm{ff}}$ is the hidden expansion width, $\sigma$ is an [[concepts/architectures/activation-function|activation function]], and $Y\in\mathbb{R}^{T\times d}$ preserves the residual dimension.

## Transformer FFN

A pre-norm Transformer usually applies:

$$
X_{\mathrm{out}} = X + \operatorname{FFN}(\operatorname{LN}(X))
$$

The FFN does not mix tokens by itself. Token mixing comes from [[concepts/architectures/attention|attention]], recurrence, convolution, or message passing. The FFN changes the representation at each token after information has been mixed.

## Parameter and Compute Scale

For hidden width $d$ and expansion width $d_{\mathrm{ff}}$, a two-layer FFN has approximately:

$$
2dd_{\mathrm{ff}} + d_{\mathrm{ff}} + d
$$

parameters including biases. If $d_{\mathrm{ff}}=4d$, the FFN parameter count is roughly:

$$
8d^2
$$

per block. This is often a large fraction of Transformer parameters and compute.

For a sequence length $T$, token-wise FFN cost scales as:

$$
O(Tdd_{\mathrm{ff}})
$$

unlike attention's dense token-mixing cost, which often includes an $O(T^2d)$ term.

## Gated FFN

Many modern models use a gated variant:

$$
H = \sigma(XW_g + b_g) \odot (XW_u + b_u)
$$

$$
Y = HW_o + b_o
$$

Here $\odot$ is elementwise multiplication. The gate controls which hidden features pass through, linking FFNs to [[concepts/architectures/gating|gating]] and mixture-style computation.

Common gated variants include GLU-style, GEGLU-style, and SwiGLU-style FFNs. The paper note [[papers/architectures/glu-variants-improve-transformer|GLU Variants Improve Transformer]] is the canonical reference in this wiki for reading those variants as Transformer architecture blocks:

$$
\operatorname{SwiGLU}(x)
=
\operatorname{SiLU}(xW_g)\odot xW_u
$$

The gate changes both capacity and parameter count, so comparisons should state the expansion ratio and whether the total parameter budget is matched.

## Role in Different Architectures

| Architecture | FFN Role | What Mixes Positions |
| --- | --- | --- |
| Transformer | token-wise channel mixing after attention | self-attention |
| MLP-Mixer style model | channel mixing and sometimes token mixing through separate MLPs | token-mixing MLP |
| Graph network | node-wise update after aggregation | message passing |
| CNN/ViT hybrid | channel mixing after spatial mixing | convolution or attention |
| MoE model | sparse expert FFNs | router selects experts |

The phrase "MLP block" can mean channel-only mixing or token mixing depending on the architecture. State the axis.

## Canonical Papers

| Paper | Why It Matters |
| --- | --- |
| [MLP-Mixer](/papers/architectures/mlp-mixer) | makes token-axis and channel-axis MLP mixing explicit in a vision backbone |
| [GLU Variants Improve Transformer](/papers/architectures/glu-variants-improve-transformer) | canonical note for gated Transformer FFN variants |

## Checks

- Does the block mix tokens, or only transform each token independently?
- What is the expansion ratio $d_{\mathrm{ff}}/d$?
- Is the activation ReLU, GELU, SiLU, GLU, or another gated form?
- Does the output dimension match the [[concepts/architectures/residual-connection|residual connection]]?
- Is dropout, normalization, or routing applied inside the FFN?
- Is parameter count matched when comparing gated, MoE, or wider FFNs?
- Which axis is mixed: channel, token, node, spatial location, or expert route?

## Related

- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/mixture-of-experts|Mixture of experts]]
- [[papers/architectures/mlp-mixer|MLP-Mixer]]
- [[papers/architectures/glu-variants-improve-transformer|GLU Variants Improve Transformer]]
