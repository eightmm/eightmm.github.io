---
title: Multilayer Perceptrons
tags:
  - architectures
  - mlp
  - neural-networks
---

# Multilayer Perceptrons

Multilayer perceptrons map fixed-size inputs through stacked dense layers and nonlinearities. They are simple baselines, projection heads, and feed-forward blocks inside larger architectures.

One layer is:

$$
h_{\ell+1} = \sigma(W_\ell h_\ell + b_\ell)
$$

Stacking these affine transforms and nonlinearities gives the full MLP.

In this expression, $W_\ell h_\ell + b_\ell$ is a [[concepts/architectures/linear-layer|linear layer]] and $\sigma$ is an [[concepts/architectures/activation-function|activation function]].

## Key Ideas

- Each layer applies a learned affine transform followed by a nonlinearity.
- MLPs assume a fixed-size vector input and do not directly encode order, locality, or graph structure.
- They are strong baselines when features already contain the relevant structure.
- Larger architectures use MLPs as feed-forward blocks, projection heads, readouts, and small adapters.
- Normalization, residual connections, dropout, and activation choice often matter more than the label "MLP" suggests.

## Practical Checks

- Check what features enter the MLP and whether they leak target information.
- Compare against simple linear or shallow baselines before attributing gains to architecture depth.
- Watch input scaling, missing values, and categorical encoding for tabular settings.
- For representation learning, check whether the MLP head is used only during training or also at evaluation.

## Related

- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/learning/index|Learning methods]]
