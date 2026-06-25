---
title: Augmentation Policy
tags:
  - learning
  - self-supervised-learning
  - data
---

# Augmentation Policy

An augmentation policy defines how raw examples are transformed into alternative views during training. In self-supervised learning, the policy often defines the invariances a representation is expected to learn.

For an input $x$, two views can be sampled as:

$$
v_1 \sim a_1(x),
\qquad
v_2 \sim a_2(x)
$$

where $a_1$ and $a_2$ are stochastic transformations such as masking, cropping, noise injection, graph perturbation, sequence masking, or coordinate transforms.

## Why It Matters

- Augmentation defines what information should be preserved.
- Strong augmentations can improve invariance but destroy label-relevant content.
- Weak augmentations can make the pretext task too easy.
- Domain-specific data needs domain-specific validity checks.

## Examples

- Image: crop, color jitter, blur, patch masking.
- Text or sequence: token masking, span masking, dropout, corruption.
- Graph: node masking, edge dropping, subgraph sampling.
- Molecule: conformer perturbation, atom masking, graph masking, view generation that preserves chemical identity.
- 3D structure: rigid transform, coordinate noise, residue masking, distance/contact masking.

## Checks

- Does the transformation preserve the downstream label or target semantics?
- Does it create unrealistic examples?
- Does it leak the answer through a view construction rule?
- Are positives still positive after augmentation?
- Are false negatives likely because different examples share the same class, scaffold, family, or target?

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/representation-collapse|Representation collapse]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/evaluation/leakage|Leakage]]
