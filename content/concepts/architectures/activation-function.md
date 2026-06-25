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

## Checks

- Check whether the activation is ReLU, GELU, SiLU, tanh, sigmoid, or gated.
- Watch saturation for sigmoid/tanh.
- In generative models, distinguish hidden activations from output distributions.
- In scientific models, ensure output activation matches the target range.

## Related

- [[concepts/architectures/mlp|MLP]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/machine-learning/optimization|Optimization]]
