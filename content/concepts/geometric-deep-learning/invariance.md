---
title: Invariance
tags:
  - geometric-deep-learning
  - symmetry
  - invariance
---

# Invariance

Invariance means a model output stays unchanged when an allowed transformation is applied to the input.

Formally:

$$
f(g\cdot x) = f(x)
$$

Invariant outputs are appropriate for scalar targets such as class labels, many energies, or global scores.

## Why It Matters

- Molecular energies, class labels, and many scores should not change under global rotation or translation.
- Invariant features can summarize geometry without depending on an arbitrary coordinate frame.
- Choosing invariant outputs requires knowing the task target and its physical meaning.

## Checks

- Is the target truly invariant under rotation, translation, permutation, or reflection?
- Are invariant features discarding information needed for an equivariant output?
- Does pooling respect graph membership and atom or residue identity?
- Is invariance produced by architecture, augmentation, or preprocessing?

## Related

- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/geometric-deep-learning/equivariance|Equivariance]]
- [[concepts/geometric-deep-learning/e3|E(3)]]
- [[concepts/geometric-deep-learning/se3|SE(3)]]
- [[entities/structure|Structure]]
