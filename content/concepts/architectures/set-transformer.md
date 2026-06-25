---
title: Set Transformer
tags:
  - architectures
  - set-modeling
  - attention
---

# Set Transformer

Set Transformers apply attention to unordered sets. They extend [[concepts/architectures/deep-sets|Deep Sets]] by allowing pairwise or higher-order interactions between elements while preserving permutation symmetry.

For set elements $X=\{x_i\}_{i=1}^{n}$, self-attention over elements can be written as:

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

$$
Y=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Because the same projections and attention rule are applied to every element, the output is permutation equivariant. A final pooling or learned seed attention step can produce permutation-invariant set outputs.

## Key Ideas

- Deep Sets pool independent element embeddings; Set Transformers let elements interact before readout.
- Attention cost grows with the number of set elements, so inducing points or latent bottlenecks are often used.
- They are useful for candidate sets, point sets, retrieved contexts, molecular conformer sets, and multi-instance inputs.
- The distinction from sequence Transformers is that element order should not carry meaning unless explicit positional features are added.

## Practical Checks

- Decide whether the target should be permutation invariant or equivariant.
- Check whether positional, rank, or index features accidentally introduce order dependence.
- Watch set size scaling and masking for variable-size sets.
- Compare against [[concepts/architectures/graph-construction|graph construction]] when relationships are sparse or typed.

## Related

- [[concepts/architectures/deep-sets|Deep Sets]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/perceiver|Perceiver]]
- [[concepts/geometric-deep-learning/invariance|Invariance]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
