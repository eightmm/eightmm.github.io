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

## Policy Choices

- Canonicalize: choose one deterministic representative for deduplication and splitting.
- Preserve: keep supplied tautomer states when the assay or structure supports them.
- Enumerate: generate a tautomer set and score or aggregate over it.
- Condition: choose tautomer state jointly with pH, protonation, binding site, or preparation protocol.

If a model uses one representative, the identity key should include the protocol:

$$
k = H(\operatorname{canon}_{\pi}(m))
$$

where $\pi$ is the tautomer canonicalization policy.

## Interaction With Protonation

Tautomer and protonation decisions are coupled:

$$
\text{state}(m) = (\text{tautomer}, \text{protonation}, \text{charge})
$$

Changing one can change hydrogen bonding, formal charge, conformers, and docking interactions.

## Checks

- Is tautomer canonicalization applied before splitting?
- Is the chosen tautomer protocol recorded?
- Could the target distinguish tautomers biologically?
- Are 3D conformers generated after tautomer/protonation decisions?
- Are labels tied to a molecule definition that matches the model input?
- Does the featurizer cache key change when the tautomer policy changes?

## Failure Modes

- Equivalent tautomer records appear on both sides of a split.
- Over-canonicalization removes a tautomer state that the target distinguishes.
- Docking uses a different tautomer/protonation state than the 2D featurizer.
- Generated molecules are evaluated before tautomer normalization, inflating novelty.
- Label aggregation silently averages measurements from different state definitions.

## Related

- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/molecular-modeling/protonation-state|Protonation state]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/evaluation/leakage|Leakage]]
