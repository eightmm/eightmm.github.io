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

## Checks

- Is the SMILES canonical, randomized, or enumerated?
- Are stereochemistry, aromaticity, charges, isotopes, and salts represented?
- Was molecular standardization performed before deduplication and splitting?
- Does tokenization preserve bracket atoms and multi-character symbols?
- Are invalid strings filtered, repaired, or reported?

## Related

- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[entities/molecule|Molecule]]
