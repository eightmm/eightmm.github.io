---
title: Tautomer
tags:
  - molecular-modeling
  - chemistry
---

# Tautomer

A tautomer is an alternative molecular form created by moving atoms and bonds, often involving hydrogen relocation and bond-order changes. Tautomers can share the same formula but differ in geometry, charge distribution, and interaction pattern.

A raw molecule can map to a tautomer set:

$$
\mathcal{T}(m) = \{m_1, m_2, \ldots, m_k\}
$$

A canonicalization protocol chooses one representative:

$$
\tilde{m} = \operatorname{canon}(\mathcal{T}(m))
$$

## Why It Matters

Tautomer handling changes deduplication, scaffold construction, conformer generation, docking, and property labels. A silent tautomer policy can make two pipelines look comparable when they are not.

## Checks

- Is tautomer canonicalization applied before splitting?
- Is the chosen tautomer protocol recorded?
- Could the target distinguish tautomers biologically?
- Are 3D conformers generated after tautomer/protonation decisions?
- Are labels tied to a molecule definition that matches the model input?
- Does the featurizer cache key change when the tautomer policy changes?

## Related

- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/molecular-modeling/protonation-state|Protonation state]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/evaluation/leakage|Leakage]]
