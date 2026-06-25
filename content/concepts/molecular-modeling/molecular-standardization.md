---
title: Molecular Standardization
tags:
  - molecular-modeling
  - data
  - chem-bio-ml
---

# Molecular Standardization

Molecular standardization converts raw molecular records into a consistent representation before deduplication, splitting, featurization, or modeling. It is the implementation side of a [[concepts/molecular-modeling/molecular-identity|Molecular identity]] policy.

A practical pipeline is:

$$
\tilde{m}
= S(m)
= S_{\mathrm{tautomer}}
\circ S_{\mathrm{charge}}
\circ S_{\mathrm{fragment}}
\circ S_{\mathrm{cleanup}}(m)
$$

$m$ is the raw molecule, $S$ is the standardization protocol, and $\tilde{m}$ is the standardized molecule used for hashing and downstream processing.

## Why It Matters

The same chemical entity can appear as salts, counterions, different tautomers, different protonation states, or inconsistent stereochemical records. If raw records are split before standardization, near-duplicates can leak across train and test.

## Decisions

- Remove salts or keep formulation-specific records.
- Preserve or flatten stereochemistry.
- Canonicalize tautomers or keep state-specific forms.
- Assign [[concepts/molecular-modeling/protonation-state|protonation state]] with a documented protocol.
- Hash standardized molecules before deduplication and split construction.
- Keep the standardized identity key with downstream labels and features.

## Ordering

Standardization should happen before deduplication and split assignment:

$$
m
\xrightarrow{S}
\tilde{m}
\xrightarrow{H}
\text{split key}
$$

Splitting on raw records can place equivalent standardized molecules on both sides of the split, creating leakage even when row IDs are disjoint.

## Checks

- Is standardization done before deduplication?
- Is the standardized form used as the split key?
- Is the identity policy strong enough for the task but not over-collapsing meaningful states?
- Are failures in parsing or sanitization logged instead of silently dropped?
- Are tautomer, charge, salt, and stereo choices recorded?
- Could over-standardization collapse biologically meaningful differences?
- Does the downstream featurizer consume the standardized form rather than the raw record?

## Related

- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/molecular-modeling/molecular-identity|Molecular identity]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/tautomer|Tautomer]]
- [[concepts/molecular-modeling/protonation-state|Protonation state]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/leakage|Leakage]]
