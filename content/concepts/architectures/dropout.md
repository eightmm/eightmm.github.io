---
title: Dropout
tags:
  - architectures
  - regularization
---

# Dropout

Dropout is a regularization method that randomly removes activations during training. It discourages a model from relying too heavily on any single hidden unit or path.

With drop probability $p$ and keep probability $q=1-p$:

$$
m_i \sim \operatorname{Bernoulli}(q)
$$

$$
\tilde{h}
= \frac{m \odot h}{q}
$$

The scaling keeps the expected activation size similar during training.

The expectation is:

$$
\mathbb{E}[\tilde{h}_i]
=
\mathbb{E}\left[\frac{m_i h_i}{q}\right]
=
h_i
$$

At inference time, standard inverted dropout is disabled:

$$
h_{\mathrm{test}} = h
$$

because the training-time scaling already preserves the expected activation.

## Common Variants

- Feature dropout: randomly masks hidden activations.
- Attention dropout: masks attention weights or attention probabilities.
- Residual dropout: applies dropout on a residual branch output.
- Token or patch dropout: removes input tokens, patches, or graph elements.
- DropPath or stochastic depth: drops whole residual branches.

The variant matters because dropping a feature, token, attention edge, or residual path changes different information routes.

## Where It Appears

- Classifier heads for small labeled datasets.
- Transformer attention and feed-forward blocks.
- CNN and ViT regularization through dropout, stochastic depth, or token dropping.
- Graph and molecular models through node, edge, feature, or subgraph dropout.

## Checks

- Dropout is usually active during training and disabled during inference.
- Too much dropout can underfit.
- Placement matters: attention dropout, residual dropout, feature dropout, and classifier dropout are not identical.
- In small scientific datasets, dropout is only one part of regularization.
- Check whether reported results use deterministic inference or Monte Carlo dropout.
- Check that dropout is disabled during final evaluation unless uncertainty sampling is explicitly part of the method.
- For graph, sequence, and structure data, confirm that dropped elements do not destroy label-defining information.

## Related

- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
