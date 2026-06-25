---
title: Molecular Identity
tags:
  - molecular-modeling
  - chem-bio-ml
  - data
---

# Molecular Identity

Molecular identity defines what counts as "the same molecule" in a dataset. This decision controls deduplication, split construction, featurization, label aggregation, and leakage checks.

A practical identity key is not only the raw string:

$$
I(m)
=
H(
\operatorname{standardize}(m),
\operatorname{stereo},
\operatorname{tautomer},
\operatorname{charge},
\operatorname{salt\ policy}
)
$$

where $H$ is a hash or canonical key used for deduplication and splitting.

## Why It Matters

The same compound can appear as:

- different SMILES strings
- salts or counterion forms
- different tautomer records
- different protonation states
- missing or conflicting stereochemistry
- duplicated records across assays or sources

If identity is defined too early from raw records, equivalent compounds can cross train/test boundaries. If identity is over-standardized, biologically meaningful stereoisomers, tautomers, or charge states can collapse into one object.

## Identity Levels

Different tasks need different identity policies:

- Connectivity identity: same atom-bond graph, ignoring stereo and state.
- Stereo-aware identity: graph plus stereochemical specification.
- State-aware identity: graph, stereo, tautomer, charge, and protonation protocol.
- Formulation-aware identity: includes salts, counterions, or formulation details.
- Assay-aware identity: molecule identity plus target, assay, endpoint, and unit.

For structure-based modeling, state-aware identity is usually safer because [[concepts/molecular-modeling/protonation-state|Protonation state]], [[concepts/molecular-modeling/tautomer|Tautomer]], and [[concepts/molecular-modeling/stereochemistry|Stereochemistry]] can change the 3D pose and interaction pattern.

## Split Key

Deduplication and split assignment should use the same identity definition:

$$
m_{\mathrm{raw}}
\xrightarrow{\text{standardize}}
\tilde{m}
\xrightarrow{\text{identity key}}
I(\tilde{m})
\xrightarrow{\text{split policy}}
\{\text{train},\text{validation},\text{test}\}
$$

If the split key is weaker than the featurized identity, leakage can remain. If the split key is stronger than the label identity, replicate labels can be split apart and inflate uncertainty.

## Label Aggregation

When multiple measurements map to the same identity:

$$
\{y_1,\ldots,y_k\}
\rightarrow
\tilde{y}
$$

the aggregation rule must be explicit: median, mean, keep-by-assay, drop conflicts, or model censored values separately. This connects molecular identity to [[entities/target-assay-label|Target-assay-label contract]] and [[concepts/evaluation/assay-harmonization|Assay harmonization]].

## Checks

- Is the molecular identity policy stated before deduplication?
- Are salts, stereo, tautomers, charge, and protonation handled explicitly?
- Is the identity key used for split construction and leakage checks?
- Are duplicate labels aggregated or kept by assay with a stated policy?
- Could the identity policy collapse molecules that the target distinguishes?
- Does the featurizer cache key include the identity and featurizer version?

## Related

- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/molecular-modeling/tautomer|Tautomer]]
- [[concepts/molecular-modeling/protonation-state|Protonation state]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/leakage|Leakage]]
- [[entities/bioactivity-label|Bioactivity label]]
