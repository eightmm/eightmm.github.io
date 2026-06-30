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

Stacking these affine transforms and nonlinearities gives the full MLP:

$$
f_\theta(x)
=
W_L h_{L-1}+b_L,
\qquad
h_0=x,
\qquad
h_\ell=\sigma_\ell(W_\ell h_{\ell-1}+b_\ell)
$$

In this expression, $W_\ell \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$, $b_\ell \in \mathbb{R}^{d_\ell}$, and $\sigma_\ell$ is an [[concepts/architectures/activation-function|activation function]]. The final layer may be followed by a task-specific readout such as [[concepts/architectures/softmax|Softmax]] for classification or an identity map for regression.

The parameter count is roughly:

$$
|\theta|=\sum_{\ell=1}^{L}(d_\ell d_{\ell-1}+d_\ell)
$$

This is why width and depth change both capacity and memory cost even when the input representation stays fixed.

## Key Ideas

- Each layer applies a learned affine transform followed by a nonlinearity.
- MLPs assume a fixed-size vector input and do not directly encode order, locality, or graph structure.
- They are strong baselines when features already contain the relevant structure.
- Larger architectures use MLPs as feed-forward blocks, projection heads, readouts, and small adapters.
- Normalization, residual connections, dropout, and activation choice often matter more than the label "MLP" suggests.

## Where MLPs Appear

| Role | Example |
| --- | --- |
| Baseline model | tabular features, fingerprints, pooled embeddings |
| Projection head | contrastive learning, JEPA-style representation targets |
| Feed-forward block | Transformer FFN, MLP-Mixer channel mixing |
| Readout head | graph-level or sequence-level scalar prediction |
| Adapter | small trainable module on top of frozen embeddings |

## Canonical Papers

| Paper | Why It Matters |
| --- | --- |
| [MLP-Mixer](/papers/architectures/mlp-mixer) | separates token-mixing and channel-mixing MLPs in a vision backbone |
| [GLU Variants Improve Transformer](/papers/architectures/glu-variants-improve-transformer) | shows how gated MLP variants change Transformer feed-forward blocks |

## Practical Checks

- Check what features enter the MLP and whether they leak target information.
- Compare against simple linear or shallow baselines before attributing gains to architecture depth.
- Watch input scaling, missing values, and categorical encoding for tabular settings.
- For representation learning, check whether the MLP head is used only during training or also at evaluation.
- State whether the claim is about the backbone representation, the MLP head, or the full pipeline.

## Related

- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/learning/index|Learning methods]]
