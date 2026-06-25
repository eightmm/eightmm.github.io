---
title: Molecular Similarity
tags:
  - molecular-modeling
  - similarity
  - cheminformatics
---

# Molecular Similarity

Molecular similarity measures how close two molecules are under a chosen representation. It is used for retrieval, clustering, scaffold analysis, activity cliffs, nearest-neighbor baselines, and diversity checks.

For binary fingerprints $a,b\in\{0,1\}^d$, the Tanimoto similarity is:

$$
T(a,b)
=
\frac{a\cdot b}
{\lVert a\rVert_1+\lVert b\rVert_1-a\cdot b}
$$

The value ranges from 0 to 1 for binary fingerprints, with 1 meaning identical fingerprints.

## Key Ideas

- Similarity depends on representation: fingerprints, graphs, descriptors, conformers, and 3D shapes can disagree.
- High fingerprint similarity does not guarantee same binding pose or activity.
- Low similarity does not prove different mechanism if the representation misses relevant features.
- Similarity baselines are important before claiming a deep model learns new chemistry.

## Practical Checks

- What representation defines similarity?
- Is similarity computed before or after molecule standardization?
- Are stereochemistry, protonation, tautomers, and salts handled consistently?
- Is the threshold chosen for retrieval, clustering, splitting, or diversity?
- Does a simple nearest-neighbor baseline explain the reported performance?

## Related

- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/evaluation/activity-cliff|Activity cliff]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/tasks/retrieval|Retrieval]]
