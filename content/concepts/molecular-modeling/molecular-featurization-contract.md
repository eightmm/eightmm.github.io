---
title: Molecular Featurization Contract
tags:
  - molecular-modeling
  - feature-engineering
  - data
---

# Molecular Featurization Contract

A molecular featurization contract defines how a standardized molecule becomes a model input. It prevents train, evaluation, and inference from using subtly different molecule definitions.

A featurizer is a versioned function:

$$
z = F_\psi(\tilde{m})
$$

where $\tilde{m}$ is the standardized molecule, $F_\psi$ is the featurizer with versioned parameters $\psi$, and $z$ is a fingerprint, graph, descriptor vector, conformer, or learned representation.

## Contract Fields

| Field | Record |
| --- | --- |
| Input identity | standardized molecule hash and standardization protocol |
| Chemical state | tautomer, protonation, charge, salt/counterion, stereochemistry |
| Representation type | SMILES, graph, fingerprint, descriptor, conformer, 3D pose, or learned embedding |
| Feature definition | atom features, bond features, fingerprint radius/size, descriptor list, conformer protocol |
| Software version | toolkit, model, featurizer code, and random seed when applicable |
| Failure policy | parse failure, sanitization failure, missing stereo, invalid valence, conformer failure |
| Cache key | input hash plus featurizer version |

## Cache Identity

Feature cache keys should include both molecule identity and featurizer identity:

$$
k = H(\tilde{m}, \psi)
$$

If either $\tilde{m}$ or $\psi$ changes, cached features are invalid.

## Representation Types

| Representation | Good For | Main Caveat |
| --- | --- | --- |
| canonical SMILES | sequence models, exact string pipelines | tokenization and canonicalization policy can change the input |
| randomized SMILES | augmentation and sequence robustness | train/test augmentation policy must be fixed |
| molecular graph | GNNs and atom/bond inductive bias | aromaticity, bond order, charge, stereo, and hydrogens must be explicit |
| fingerprint | fast baselines, similarity, retrieval | radius, bit length, collision, and count/binary choice matter |
| descriptors | small data baselines, interpretability | descriptor list and missing-value policy can leak preprocessing choices |
| conformer or coordinates | geometry, docking, 3D generation | conformer source and deployment availability define the claim |
| learned embedding | retrieval or downstream prediction | frozen vs trainable encoder changes evaluation meaning |

## Leakage Risks

Featurization can leak information even when labels are not directly used:

$$
F_\psi(\tilde{m})
\not\perp
\text{test protocol}
$$

when conformers, filters, standardization statistics, vocabulary, or embedding models are fit using test data or deployment-unavailable context.

Common risks:

- conformer or pose generated with knowledge of the bound ligand or target when inference will not have it;
- descriptors normalized using full-dataset statistics before the split;
- duplicate handling after featurization rather than before the split;
- fingerprint or embedding cache reused after changing standardization;
- invalid molecules silently dropped differently across train and test;
- scaffold split computed on a different molecular identity than the model input.

## Checks

- Is the same featurizer used at train, validation, test, and inference time?
- Are invalid molecules logged rather than silently dropped?
- Are stereochemistry and protonation handled consistently with the task?
- Does featurization distinguish molecules that the label distinguishes?
- Does graph batching keep molecules isolated?
- Does a simple fingerprint baseline already explain the benchmark?
- Is the split computed on the same standardized identity used by the featurizer?
- Are conformer and coordinate features available under the intended inference setting?

## Related

- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/chemical-state-contract|Chemical state contract]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/modalities/representation-contract|Representation contract]]
- [[concepts/evaluation/leakage|Leakage]]
