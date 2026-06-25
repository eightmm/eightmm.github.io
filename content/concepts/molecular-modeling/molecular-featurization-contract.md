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

- Input identity: standardized molecule hash and standardization protocol.
- Chemical state: tautomer, protonation, charge, salt/counterion, stereochemistry.
- Representation type: SMILES, graph, fingerprint, descriptor, conformer, or 3D pose.
- Feature definition: atom features, bond features, fingerprint radius/size, descriptor list, or conformer protocol.
- Software version: toolkit, model, featurizer code, and random seed when applicable.
- Failure policy: parse failure, sanitization failure, missing stereo, invalid valence, or conformer failure.
- Cache key: input hash plus featurizer version.

## Cache Identity

Feature cache keys should include both molecule identity and featurizer identity:

$$
k = H(\tilde{m}, \psi)
$$

If either $\tilde{m}$ or $\psi$ changes, cached features are invalid.

## Checks

- Is the same featurizer used at train, validation, test, and inference time?
- Are invalid molecules logged rather than silently dropped?
- Are stereochemistry and protonation handled consistently with the task?
- Does featurization distinguish molecules that the label distinguishes?
- Does graph batching keep molecules isolated?
- Does a simple fingerprint baseline already explain the benchmark?

## Related

- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/evaluation/leakage|Leakage]]
