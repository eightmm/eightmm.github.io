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

For bit vectors, the same formula is:

$$
\operatorname{sim}_{\mathrm{Tanimoto}}(a,b)
=
\frac{a^\top b}
{\lVert a\rVert_1+\lVert b\rVert_1-a^\top b}
$$

## ECFP View

Circular fingerprints such as ECFP hash local atom environments up to a radius $r$ into a fixed-length vector:

$$
f(M)
=
H\left(
\bigcup_{v\in V(M)}
\operatorname{Env}_r(v)
\right)
$$

where $V(M)$ is the atom set and $\operatorname{Env}_r(v)$ is the neighborhood around atom $v$. This makes fingerprints strong for local substructure similarity, but weaker for 3D conformation, long-range geometry, and protein-context-dependent binding.

## Count vs Binary

Binary fingerprints store whether a feature appears:

$$
f_j(M)\in\{0,1\}
$$

Count fingerprints store how often it appears:

$$
f_j(M)\in\mathbb{N}
$$

The choice changes similarity and model behavior. Count features can matter for repeated motifs, while binary features are often robust and compact.

## Fingerprint Contract

A reproducible fingerprint is defined by settings, not only by a name.

| Field | Examples |
| --- | --- |
| molecule state | standardized, tautomer policy, protonation, stereo |
| algorithm | Morgan/ECFP, MACCS, path-based, pharmacophore |
| radius/depth | local environment size |
| bit size | 1024, 2048, 4096, sparse vector |
| chirality | included, ignored, enumerated |
| count mode | binary bit, count vector, sparse count |
| toolkit/version | RDKit version and generator settings |

Without this contract, “ECFP” is underspecified.

## Collisions

Hashed fingerprints can map different substructures to the same bit:

$$
H(a)=H(b),\quad a\neq b
$$

Larger bit sizes reduce collision probability but increase memory and compute. Because RDKit versions and fingerprint settings can change behavior, the featurizer contract should record radius, bit size, chirality, feature flags, and software version.

## Fingerprints and Splits

Fingerprints are often used both as input features and as split diagnostics. Keep those roles separate:

| Role | Use | Risk |
| --- | --- | --- |
| model input | property prediction, screening model | benchmark may reward memorized chemistry |
| similarity baseline | nearest-neighbor or kernel method | baseline omitted or under-tuned |
| clustering | diversity or scaffold-like grouping | clusters depend on threshold and representation |
| leakage audit | detect near-duplicates across splits | misses non-fingerprint leakage |

If a split is built with a fingerprint similarity threshold, report that threshold and representation beside the benchmark metric.

## Baseline Role

A fingerprint baseline is a serious baseline, not a toy. For molecular property prediction or virtual screening, a model should be compared against:

$$
\text{standardized molecule}
\rightarrow
\text{fingerprint}
\rightarrow
\text{RF/GBM/linear model}
$$

If a deep model does not beat this baseline on a domain-appropriate split, the architecture is probably not adding useful signal.

## Checks

- What fingerprint type, radius, bit size, and feature definition are used?
- Are chirality and bond features included?
- Is the fingerprint version pinned and reproducible?
- Does a simple fingerprint baseline already solve the benchmark?
- Are scaffold or cluster splits based on standardized molecules?
- Is the fingerprint computed after standardization and before split leakage checks?
- Is the same fingerprint contract used for train and inference?
- Are activity cliffs and near-neighbor errors inspected separately?
- Is a property-only or fingerprint-only decoy baseline exposing benchmark bias?
- Are fingerprint settings identical across training, validation, test, and inference?
- Is the fingerprint used for both input features and split construction disclosed?

## Related

- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/molecular-similarity|Molecular similarity]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/activity-cliff|Activity cliff]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[entities/molecule|Molecule]]
