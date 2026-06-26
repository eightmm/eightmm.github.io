---
title: SMILES
tags:
  - molecular-modeling
  - molecules
  - sequence
---

# SMILES

SMILES is a line notation that represents molecular graphs as strings. It is convenient for storage and sequence models, but one molecule can have many valid SMILES strings unless canonicalization or augmentation is controlled.

A tokenized SMILES input can be written as:

$$
t_{1:L}
= \operatorname{Tokenizer}(\operatorname{SMILES}(M))
$$

where $M$ is a molecule and $t_{1:L}$ are string tokens.

## Canonical vs Randomized

A canonical SMILES function maps a standardized molecule to one deterministic string:

$$
s_{\mathrm{canon}}
=
C(\tilde{M})
$$

where $\tilde{M}$ is the molecule after [[concepts/molecular-modeling/molecular-standardization|molecular standardization]]. Canonical strings are useful for deduplication, hashing, caching, and reproducible train/test splits.

Randomized or enumerated SMILES sample one of many valid traversals:

$$
s
\sim
q(s\mid \tilde{M})
$$

This can be useful as data augmentation for sequence models, but it should not change the molecular identity or label. The split key should still be the standardized molecule, not the enumerated string.

## Tokenization

SMILES tokenization is not ordinary character tokenization. Multi-character atoms and bracket expressions must be preserved:

$$
\mathrm{Cl},\ \mathrm{Br},\ [\mathrm{NH}_3^+],\ [\mathrm{C@@H}]
$$

If a tokenizer splits these incorrectly, the sequence model sees invalid chemistry even when the original string was valid.

## Chemistry Carried by SMILES

SMILES can encode:

- Atoms, bonds, branches, and ring closures.
- Aromaticity conventions.
- Formal charge.
- Isotopes.
- Stereochemistry.
- Disconnected fragments such as salts or mixtures.

These fields are not only formatting details. They can change molecular identity, featurization, and biological activity.

## Modeling Use

For a SMILES language model:

$$
p_\theta(s)
=
\prod_{t=1}^{L}
p_\theta(s_t\mid s_{<t})
$$

The model learns a distribution over strings. A valid string distribution is not automatically a good molecular distribution; validity, uniqueness, novelty, property control, and standardization all need separate checks.

## Checks

- Is the SMILES canonical, randomized, or enumerated?
- Are stereochemistry, aromaticity, charges, isotopes, and salts represented?
- Was molecular standardization performed before deduplication and splitting?
- Does tokenization preserve bracket atoms and multi-character symbols?
- Are invalid strings filtered, repaired, or reported?
- Is the split key based on standardized molecular identity rather than raw SMILES?
- Are generated strings standardized before novelty, duplicate, or property evaluation?
- Are stereo and protonation choices aligned with the downstream task?

## Related

- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/molecular-identity|Molecular identity]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/molecular-modeling/protonation-state|Protonation state]]
- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[entities/molecule|Molecule]]
