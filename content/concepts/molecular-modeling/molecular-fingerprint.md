---
title: Molecular Fingerprint
tags:
  - molecular-modeling
  - feature-engineering
---

# Molecular Fingerprint

A molecular fingerprint is a fixed-size representation of molecular substructures. Fingerprints are strong classical baselines for similarity search, property prediction, and virtual screening.

A binary fingerprint is:

$$
f(M)\in\{0,1\}^{d}
$$

Similarity is often measured with Tanimoto similarity:

$$
\operatorname{sim}_{\mathrm{Tanimoto}}(a,b)
=
\frac{|a\cap b|}
{|a\cup b|}
$$

for binary fingerprints $a$ and $b$.

## Checks

- What fingerprint type, radius, bit size, and feature definition are used?
- Are chirality and bond features included?
- Is the fingerprint version pinned and reproducible?
- Does a simple fingerprint baseline already solve the benchmark?
- Are scaffold or cluster splits based on standardized molecules?

## Related

- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[entities/molecule|Molecule]]
