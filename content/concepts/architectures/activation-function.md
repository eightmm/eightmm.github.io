---
title: Activation Function
tags:
  - architectures
  - neural-networks
---

# Activation Function

An activation function adds nonlinearity between linear transformations. Without nonlinear activations, stacked linear layers collapse into a single linear layer.

For an MLP layer:

$$
h_{\ell+1} = \sigma(h_\ell W_\ell + b_\ell)
$$

Here $\sigma$ is the activation function.

If every layer were linear:

$$
h_2
=
(xW_1 + b_1)W_2 + b_2
=
x(W_1W_2) + (b_1W_2 + b_2)
$$

so depth would not create a new nonlinear function class. Activations make the model piecewise, smooth, gated, or saturating depending on the choice.

## Common Activations

ReLU:

$$
\operatorname{ReLU}(x) = \max(0,x)
$$

GELU:

$$
\operatorname{GELU}(x)
= x\Phi(x)
$$

where $\Phi$ is the standard normal CDF. GELU is common in Transformer feed-forward blocks.

SiLU, also called swish, is:

$$
\operatorname{SiLU}(x) = x\,\sigma(x)
$$

where $\sigma(x)=1/(1+\exp(-x))$ is the sigmoid function. SiLU and gated variants are common in modern [[concepts/architectures/feed-forward-network|feed-forward networks]].

Tanh maps values into $[-1,1]$:

$$
\tanh(x)
=
\frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

Sigmoid maps values into $[0,1]$:

$$
\sigma(x)
=
\frac{1}{1+\exp(-x)}
$$

They are useful for gates and bounded outputs, but can saturate when $|x|$ is large.

## Gradient Behavior

The derivative determines how gradients pass through the layer. For ReLU:

$$
\frac{d}{dx}\operatorname{ReLU}(x)
=
\begin{cases}
1 & x>0 \\
0 & x<0
\end{cases}
$$

For sigmoid:

$$
\sigma'(x)
=
\sigma(x)(1-\sigma(x))
$$

When $\sigma(x)$ is close to 0 or 1, the derivative is small. This is one source of vanishing gradients in deep or recurrent models.

## Gated Activations

Many modern blocks use a gate:

$$
\operatorname{GLU}(a,b)
=
a \odot \sigma(b)
$$

or a smooth variant:

$$
\operatorname{SwiGLU}(a,b)
=
a \odot \operatorname{SiLU}(b)
$$

Gating lets the network modulate information flow, connecting activations to [[concepts/architectures/gating|Gating]] and [[concepts/architectures/feed-forward-network|feed-forward networks]].

## Choosing an Activation

- ReLU: simple, sparse, cheap, but can create dead units.
- GELU: smooth, common in Transformer blocks.
- SiLU/SwiGLU: smooth gated behavior, common in modern large models.
- Tanh/sigmoid: useful for gates or bounded outputs, but saturation must be watched.
- Output activation: should match the target semantics and loss.

## Checks

- Check whether the activation is ReLU, GELU, SiLU, tanh, sigmoid, or gated.
- Watch saturation for sigmoid/tanh.
- Check whether the activation is hidden-layer nonlinearity, gate, or output transform.
- Confirm whether the loss expects logits or probabilities.
- In generative models, distinguish hidden activations from output distributions.
- In scientific models, ensure output activation matches the target range.

## Related

- [[concepts/architectures/mlp|MLP]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/transformer|Transformer]]
- [[papers/architectures/gelu|GELU]]
- [[papers/architectures/glu-variants-improve-transformer|GLU Variants Improve Transformer]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/machine-learning/loss-function|Loss function]]
