---
title: RDKit
tags:
  - molecular-modeling
  - rdkit
  - cheminformatics
---

# RDKit

RDKit is a cheminformatics toolkit commonly used to parse molecules, standardize chemical records, compute descriptors and fingerprints, build molecular graphs, search substructures, and generate conformers. In this wiki, RDKit belongs under reusable molecular modeling concepts, not under paper notes.

The main role of RDKit in ML workflows is to define a deterministic transformation:

$$
m_{\mathrm{raw}}
\xrightarrow{\mathrm{RDKit\ protocol}}
(\tilde{m}, z, c)
$$

where $m_{\mathrm{raw}}$ is a raw record such as SMILES or SDF, $\tilde{m}$ is a standardized molecule, $z$ is a feature such as a graph, fingerprint, descriptor vector, or conformer, and $c$ is metadata about the protocol.

## Common Uses

| Use | Output | Main Risk |
| --- | --- | --- |
| parsing and sanitization | molecule object | silent drop or repair policy changes dataset |
| standardization | canonical identity | over-collapsing salts, tautomers, stereo, or charge |
| fingerprinting | bit/count vector | radius, bit length, chirality, and collisions change similarity |
| descriptors | numeric feature vector | missing/NaN handling and version drift |
| substructure search | match set | query semantics and aromaticity policy |
| conformer generation | 3D coordinates | seed, protonation, force field, and failure policy dominate |

## Standardization Boundary

RDKit can clean up and canonicalize molecular records, but the chemical policy is still a modeling decision. A typical pipeline is:

$$
m
\xrightarrow{\mathrm{sanitize}}
m_1
\xrightarrow{\mathrm{fragment/parent}}
m_2
\xrightarrow{\mathrm{uncharged\ or\ chosen\ charge}}
m_3
\xrightarrow{\mathrm{tautomer\ policy}}
\tilde{m}
$$

This pipeline should run before deduplication and split assignment. If split keys are built from raw records but features are built from standardized records, equivalent molecules can appear on both sides of train/test.

## Featurization Contract

For every RDKit-derived feature, record:

| Field | Example |
| --- | --- |
| input identity | canonical SMILES or standardized molecule hash |
| standardization protocol | salt, tautomer, charge, stereo policy |
| feature family | Morgan fingerprint, descriptor vector, graph, conformer |
| feature parameters | radius, bit length, count/binary, chirality flag |
| software version | RDKit version and wrapper code version |
| failure policy | parse failure, sanitization failure, conformer failure |
| cache key | hash of standardized molecule plus featurizer version |

The cache key should change whenever either molecule identity or featurizer parameters change:

$$
k = H(\tilde{m}, \psi_{\mathrm{rdkit}})
$$

## ML Checks

- Standardize before deduplication and split construction.
- Compute scaffold or cluster split on the same standardized molecule used by the model.
- Do not silently drop invalid molecules without recording denominator changes.
- State whether chirality and bond stereo are preserved.
- State fingerprint radius, bit length, count/binary mode, and chirality flag.
- State whether conformer generation is deterministic, seeded, minimized, and available at inference.
- Re-run cheap RDKit baselines when the split, standardization, or featurizer changes.

## Where It Does Not Belong

RDKit itself is not a paper result. Put reusable protocol notes here. Put a paper under [[papers/index|Papers]] only when the page is about a specific method, benchmark, or empirical claim.

## Related

- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[molecular-modeling/molecules|Molecules]]
- [[molecular-modeling/data-evaluation|Data and evaluation]]
