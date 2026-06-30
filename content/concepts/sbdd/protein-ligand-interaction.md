---
title: Protein-Ligand Interaction
tags:
  - sbdd
  - protein-ligand
  - molecular-modeling
---

# Protein-Ligand Interaction

Protein-ligand interaction describes how a ligand and a protein binding site make contacts in 3D space. It connects docking, scoring, pose quality, binding affinity, and structure-based model interpretation.

A protein-ligand complex can be represented as:

$$
C = (P, L, X_P, X_L)
$$

where $P$ is the protein or pocket, $L$ is the ligand, and $X_P$, $X_L$ are their coordinates.

Pairwise protein-ligand distances are:

$$
d_{ij}
=
\lVert x_i^{(P)} - x_j^{(L)} \rVert_2
$$

for protein atom or residue coordinate $x_i^{(P)}$ and ligand atom coordinate $x_j^{(L)}$.

## Common Interaction Types

- Steric complementarity and shape fit.
- Hydrogen bonds.
- Hydrophobic contacts.
- Salt bridges and electrostatic interactions.
- Pi-stacking or pi-cation interactions.
- Metal coordination.
- Water-mediated contacts.

## Interaction Predicate

An interaction is more than a short distance. It usually combines atom types, geometry, and chemical state:

$$
I_{ij}^{(r)}
=
\mathbf{1}
\left[
d_{ij}\le \tau_r
\land
\operatorname{type}(i,j)=r
\land
G_r(i,j)=1
\right]
$$

where $r$ is an interaction rule, $\tau_r$ is a distance threshold, and $G_r$ encodes geometry such as angle, directionality, aromatic plane, or coordination constraints.

Distance-only contacts can be useful features, but they should not be described as chemically validated interactions without rule definitions.

## Representation Contract

| Representation | Captures | Risk |
| --- | --- | --- |
| distance matrix | pairwise proximity | ignores atom type and direction |
| binary contact map | local pocket-ligand adjacency | threshold-sensitive |
| interaction fingerprint | rule-based interaction classes | depends on protonation and hydrogens |
| heterogeneous graph | atom/residue nodes and typed edges | graph construction can leak pose assumptions |
| 3D grid or point cloud | spatial field around pocket | resolution and frame dependence |
| learned pair representation | model-specific interaction features | harder to interpret chemically |

For a pair graph:

$$
G_{PL}
=
(V_P\cup V_L,\ E_{PP}\cup E_{LL}\cup E_{PL})
$$

where $E_{PL}$ defines cross protein-ligand edges. The edge rule should be stated before interpreting model attention or message passing as interaction evidence.

## Modeling Views

- Contact map between pocket atoms/residues and ligand atoms.
- Interaction fingerprint over predefined interaction rules.
- Geometric graph with protein and ligand nodes.
- Grid or point-cloud representation around the binding site.
- Learned pair representation for scoring or pose refinement.

## Pose vs Affinity Boundary

A plausible interaction pattern does not automatically imply strong binding affinity:

$$
\text{good contacts}
\not\Rightarrow
\Delta G \text{ is accurate}
$$

Pose quality asks whether the ligand geometry is plausible or close to a reference pose. Affinity asks whether the thermodynamic or assay endpoint is predicted. Scoring functions often mix these claims, so the note should separate:

| Claim | Evidence |
| --- | --- |
| contact plausibility | interaction rules, clashes, geometry checks |
| pose recovery | RMSD, symmetry-aware atom mapping, PoseBusters-style checks |
| affinity prediction | assay endpoint, unit, split, regression/ranking metric |
| virtual screening | candidate pool, enrichment, false positive handling |

## Chemical State Sensitivity

Hydrogen bonds, salt bridges, metal coordination, and charge interactions depend on preparation:

$$
I(P,L)
=
I(P_{\mathrm{state}},L_{\mathrm{state}},X_P,X_L)
$$

Changing protonation, tautomer, hydrogens, metal treatment, water retention, or side-chain states can change the interaction fingerprint even when heavy-atom coordinates look similar.

## Checks

- Are hydrogens, protonation states, waters, metals, and cofactors handled consistently?
- Is the interaction evaluated from a predicted pose or an experimental structure?
- Are contacts physically plausible, or only close in Euclidean distance?
- Is the model using interaction evidence available at deployment time?
- Are pose quality and affinity claims separated?
- Are interaction rules, thresholds, and angle constraints stated?
- Is the pocket definition independent of leaked ligand or template information?
- Are cross-edges built from predicted geometry, reference geometry, or deployment-available inputs?

## Related

- [[concepts/tasks/interaction-prediction|Interaction prediction]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[concepts/protein-modeling/binding-site|Binding site]]
- [[concepts/sbdd/pocket-definition-contract|Pocket definition contract]]
- [[concepts/sbdd/interaction-fingerprint|Interaction fingerprint]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/pose-rmsd|Pose RMSD]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/molecular-modeling/chemical-state-contract|Chemical state contract]]
