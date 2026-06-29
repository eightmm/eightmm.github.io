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

| Domain | Common transforms | Must preserve |
| --- | --- | --- |
| Image | crop, color jitter, blur, patch masking | class or target object semantics |
| Text or sequence | token masking, span masking, dropout, corruption | recoverable linguistic or sequence context |
| Graph | node masking, edge dropping, subgraph sampling | graph identity or target-relevant relation |
| Molecule | conformer perturbation, atom masking, graph masking | chemical identity, valence, stereochemistry when relevant |
| 3D structure | rigid transform, coordinate noise, residue masking, distance/contact masking | transformation contract and physical plausibility |

## Invariance Contract

An augmentation policy implicitly states an invariance:

$$
f_\theta(a(x)) \approx f_\theta(x)
\qquad
a\sim \mathcal{A}
$$

This is only desirable if the downstream target is also invariant to $a$. If $y$ changes under the transform, enforcing invariance can damage the representation.

| Transform | Safe when | Risky when |
| --- | --- | --- |
| crop | target is still visible | crop removes the causal object |
| token masking | context can infer missing token | target depends on exact token identity |
| edge dropping | graph label is robust to missing relation | relation is chemically or physically essential |
| rigid transform | target is invariant to global pose | output must rotate equivariantly |
| coordinate noise | label tolerates small perturbation | geometry validity or chirality changes |

For coordinate data, decide whether the desired behavior is invariance or equivariance:

$$
f(RX+t)=f(X)
\quad\text{or}\quad
g(RX+t)=Rg(X)+t
$$

The first is appropriate for scalar labels such as class or energy-like scores. The second is appropriate for coordinate outputs.

## Checks

- Does the transformation preserve the downstream label or target semantics?
- Does it create unrealistic examples?
- Does it leak the answer through a view construction rule?
- Are positives still positive after augmentation?
- Are false negatives likely because different examples share the same class, scaffold, family, or target?
- Is the augmentation sampled independently across train, validation, and test in a way that does not leak identity?
- Does the evaluation report performance under natural data, augmented data, or both?

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/representation-collapse|Representation collapse]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/evaluation/leakage|Leakage]]
