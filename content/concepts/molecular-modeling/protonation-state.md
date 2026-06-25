---
title: Protonation State
tags:
  - molecular-modeling
  - chemistry
  - structure-based-ai
---

# Protonation State

Protonation state describes which atoms in a molecule or binding site carry protons under a given chemical environment. It affects charge, hydrogen bonding, docking geometry, and interaction scoring.

A molecule can be viewed as a state-dependent object:

$$
m(pH) = (G, q(pH), H(pH))
$$

$G$ is the molecular graph, $q(pH)$ is the charge assignment, and $H(pH)$ is the hydrogen/protonation pattern at a stated pH or protocol.

## Why It Matters

For 2D property models, ignoring protonation may be acceptable for some endpoints. For [[concepts/sbdd/index|structure-based drug discovery]], protonation can change docking poses, interaction fingerprints, and score interpretation.

## Checks

- Is the pH or protonation protocol stated?
- Are ligand and receptor protonation handled consistently?
- Are charged states preserved through featurization?
- Are generated or standardized molecules re-protonated before 3D use?
- Is the note public-safe and free of private target preparation details?

## Related

- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/tautomer|Tautomer]]
- [[concepts/sbdd/receptor-ligand-preparation|Receptor and ligand preparation]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/sbdd/scoring-function|Scoring function]]
