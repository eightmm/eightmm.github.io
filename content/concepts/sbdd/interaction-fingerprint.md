---
title: Interaction Fingerprint
tags:
  - sbdd
  - molecular-modeling
  - protein-ligand
---

# Interaction Fingerprint

An interaction fingerprint encodes protein-ligand contacts or interaction types as a vector. It is useful for comparing poses, clustering docking results, summarizing binding modes, and building interpretable features.

For residue or atom interaction indicators:

$$
z_k
=
\mathbf{1}[\text{interaction } k \text{ is present}]
$$

The fingerprint is:

$$
z = (z_1,\ldots,z_K)
$$

Similarity between two interaction fingerprints can be computed with overlap or Tanimoto-style measures:

$$
T(z,z')
=
\frac{z\cdot z'}
{\lVert z\rVert_1+\lVert z'\rVert_1-z\cdot z'}
$$

## Key Ideas

- Interaction fingerprints summarize contacts, not full geometry.
- The feature definition depends on distance cutoffs, atom typing, residue selection, and interaction rules.
- Similar fingerprints can hide different poses if geometry is coarse.
- They are useful diagnostic features but should not replace pose quality checks.

## Practical Checks

- Which interaction types are encoded?
- Are cutoffs and atom typing rules documented?
- Are waters, metals, cofactors, and protonation states included or ignored?
- Does the fingerprint compare poses within one target or across targets?
- Is it used for interpretation, clustering, filtering, or model input?

## Related

- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/protein-modeling/binding-site|Binding site]]
