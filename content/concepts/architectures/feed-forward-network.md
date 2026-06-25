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

## Gated FFN

Many modern models use a gated variant:

$$
H = \sigma(XW_g + b_g) \odot (XW_u + b_u)
$$

$$
Y = HW_o + b_o
$$

Here $\odot$ is elementwise multiplication. The gate controls which hidden features pass through, linking FFNs to [[concepts/architectures/gating|gating]] and mixture-style computation.

## Checks

- Does the block mix tokens, or only transform each token independently?
- What is the expansion ratio $d_{\mathrm{ff}}/d$?
- Is the activation ReLU, GELU, SiLU, GLU, or another gated form?
- Does the output dimension match the [[concepts/architectures/residual-connection|residual connection]]?
- Is dropout, normalization, or routing applied inside the FFN?

## Related

- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/mixture-of-experts|Mixture of experts]]
