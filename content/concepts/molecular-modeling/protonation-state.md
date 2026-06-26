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

## Protocol Boundary

Protonation is not assigned by a generic graph alone. A public note should state the protocol:

$$
\operatorname{prot}_{\pi}(m)
\rightarrow
(m_{\mathrm{prot}}, q, H)
$$

where $\pi$ includes pH, tool, rule set, receptor context if used, and whether hydrogens are explicit.

For structure-based modeling, ligand and receptor preparation must be compatible:

$$
\text{state}_{\mathrm{complex}}
=
(\text{ligand protonation}, \text{receptor protonation}, \text{pH/protocol})
$$

Do not imply that one tool or default state is universally correct.

## Checks

- Is the pH or protonation protocol stated?
- Are ligand and receptor protonation handled consistently?
- Are charged states preserved through featurization?
- Are generated or standardized molecules re-protonated before 3D use?
- Is the note public-safe and free of private target preparation details?
- Is protonation assigned before conformer generation and docking?
- Does the featurizer encode formal charge and hydrogens as required by the task?

## Failure Modes

- A neutralized 2D molecule is used for splitting, but charged forms are used for docking.
- Formal charge is removed or ignored in graph features.
- Ligand and receptor are prepared with incompatible protonation assumptions.
- A model trained on one protonation protocol is evaluated on another without recording the shift.
- Public notes include private target-preparation details instead of generic protocol fields.

## Related

- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/molecular-modeling/tautomer|Tautomer]]
- [[concepts/sbdd/receptor-ligand-preparation|Receptor and ligand preparation]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/sbdd/scoring-function|Scoring function]]
