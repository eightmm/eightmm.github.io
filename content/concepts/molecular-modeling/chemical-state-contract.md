---
title: Chemical State Contract
tags:
  - molecular-modeling
  - chemistry
  - data
---

# Chemical State Contract

A chemical state contract records which molecular form is being modeled before features, conformers, docking, labels, or splits are defined. A molecule is not only a graph; it can include salt policy, tautomer, protonation, charge, stereochemistry, and conformer protocol.

The practical object is:

$$
m_{\mathrm{state}}
=
(G,\ s,\ \tau,\ p,\ q,\ c)
$$

where $G$ is molecular connectivity, $s$ is stereochemistry, $\tau$ is tautomer state, $p$ is protonation state, $q$ is formal or partial charge policy, and $c$ is conformer or coordinate protocol when 3D geometry is used.

## Why It Matters

Two records can share the same topology but differ in binding, activity, docking pose, or featurized representation:

$$
G(m_1)=G(m_2)
\quad\not\Rightarrow\quad
y(m_1)=y(m_2)
$$

This is common for stereoisomers, protonation states, tautomers, salts, and conformers. If a paper does not state the chemical state policy, its reported performance can be hard to interpret or reproduce.

## State Fields

| Field | Question | Route |
| --- | --- | --- |
| Connectivity | What atom-bond graph is treated as the base molecule? | [Molecular graph](/concepts/molecular-modeling/molecular-graph) |
| Salt and fragments | Are salts, counterions, mixtures, or largest fragments kept? | [Molecular standardization](/concepts/molecular-modeling/molecular-standardization) |
| Stereochemistry | Are chiral centers and double-bond stereo preserved, flattened, unknown, or enumerated? | [Stereochemistry](/concepts/molecular-modeling/stereochemistry) |
| Tautomer | Is a canonical tautomer chosen, supplied states preserved, or states enumerated? | [Tautomer](/concepts/molecular-modeling/tautomer) |
| Protonation | What pH, rule set, receptor context, and hydrogen policy define the state? | [Protonation state](/concepts/molecular-modeling/protonation-state) |
| Charge | Are formal charge, partial charge, and neutralization policies recorded? | [Molecular featurization contract](/concepts/molecular-modeling/molecular-featurization-contract) |
| Conformer | Are 3D coordinates generated, selected, minimized, or observed experimentally? | [Conformer](/concepts/molecular-modeling/conformer) |

## Pipeline Position

Chemical state should be fixed before deduplication, split construction, featurizer caching, conformer generation, and docking:

$$
m_{\mathrm{raw}}
\xrightarrow{\text{standardize}}
m_{\mathrm{state}}
\xrightarrow{\text{identity key}}
I(m_{\mathrm{state}})
\xrightarrow{\text{split/cache/featurize}}
r
$$

If the split key uses topology only but the model consumes state-aware features, near-duplicate states can leak across splits. If the model flattens chemical states that labels distinguish, signal can be destroyed.

## 2D vs 3D Boundary

For 2D property prediction, the state contract may focus on graph, stereo, tautomer, charge, and label context. For structure-based modeling, the contract must also include conformers, receptor preparation, and pose context:

$$
\text{complex state}
=
(m_{\mathrm{ligand\ state}},\ p_{\mathrm{receptor}},\ X_{\mathrm{ligand}},\ X_{\mathrm{pocket}})
$$

This separates ligand-only conformers from protein-bound poses and prevents a docking note from silently using deployment-unavailable information.

## Paper Reading Checks

- Does the paper define the molecular identity before splitting?
- Are salts, stereo, tautomer, protonation, charge, and conformer policy stated?
- Is chemical state fixed before conformer generation, docking, and featurization?
- Does the benchmark compare methods under the same state policy?
- Are generated molecules evaluated after the same standardization and state assignment as training molecules?
- Are labels aggregated across records that may have different chemical states?
- Is novelty or scaffold split computed before or after state normalization?
- Does a 3D model use ligand-only conformers, experimental structures, predicted structures, or bound poses?

## Failure Modes

| Failure | Consequence |
| --- | --- |
| Splitting raw records before standardization | equivalent compounds can appear in train and test |
| Flattening stereochemistry | active stereoisomers can collapse with inactive ones |
| Mixing tautomer/protonation protocols | docking or affinity labels become hard to compare |
| Caching features without state policy | stale features survive after standardization changes |
| Evaluating novelty before normalization | generated molecules look falsely novel |
| Using one conformer without reporting protocol | result depends on an unrecorded geometry choice |

## Related

- [[concepts/molecular-modeling/molecular-identity|Molecular identity]]
- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/molecular-modeling/tautomer|Tautomer]]
- [[concepts/molecular-modeling/protonation-state|Protonation state]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[molecular-modeling/paper-intake|Computational Biology paper intake]]
- [[molecular-modeling/data-evaluation|Data and evaluation]]
