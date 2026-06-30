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

## Standardization Contract

Standardization is a contract, not just a preprocessing script.

| Field | Record |
| --- | --- |
| parser | input format, sanitization rule, parse-failure policy |
| fragment policy | parent molecule, largest fragment, salts, mixtures |
| charge policy | neutralization, formal charge preservation, zwitterions |
| tautomer policy | canonical, enumerated, preserved, task-specific |
| stereo policy | preserve, enumerate, reject unknown, flatten intentionally |
| output key | canonical SMILES, InChIKey layer, hash, or toolkit-specific key |

If a later table stores only the standardized string, keep a route back to the raw record and source ID for audit.

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

## Label Aggregation

After standardization, several raw records may collapse into one molecule key:

$$
k = H(S(m))
$$

For labels, the pipeline must state how duplicates are aggregated:

| Situation | Safer handling |
| --- | --- |
| same endpoint, same unit, consistent values | aggregate with documented statistic |
| same endpoint, conflicting values | keep conflict flag or remove from clean benchmark |
| different assay protocols | do not merge without assay context |
| different target construct/species | keep separate target-assay-label records |
| different chemical state intentionally measured | do not over-collapse |

This is why molecular standardization links directly to [[entities/target-assay-label|Target-assay-label contract]], not only to molecule identity.

## Toolkit Boundary

Toolkits such as [[concepts/molecular-modeling/rdkit|RDKit]] implement parts of standardization, but the chemical policy is still yours. Record whether the protocol removes fragments, uncharges molecules, canonicalizes tautomers, preserves stereochemistry, and how it handles parse failures.

Do not imply that a toolkit default is a scientific standard. It is a reproducibility choice that affects identity, split keys, fingerprints, descriptors, and conformers.

## Checks

- Is standardization done before deduplication?
- Is the standardized form used as the split key?
- Is the identity policy strong enough for the task but not over-collapsing meaningful states?
- Are failures in parsing or sanitization logged instead of silently dropped?
- Are tautomer, charge, salt, and stereo choices recorded?
- Could over-standardization collapse biologically meaningful differences?
- Does the downstream featurizer consume the standardized form rather than the raw record?
- Are duplicate labels aggregated only when endpoint, unit, target, and assay context match?
- Is the standardization version included in cached feature keys?

## Related

- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/molecular-modeling/rdkit|RDKit]]
- [[concepts/molecular-modeling/molecular-identity|Molecular identity]]
- [[concepts/molecular-modeling/chemical-state-contract|Chemical state contract]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/tautomer|Tautomer]]
- [[concepts/molecular-modeling/protonation-state|Protonation state]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[concepts/data/data-schema|Data schema]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/leakage|Leakage]]
