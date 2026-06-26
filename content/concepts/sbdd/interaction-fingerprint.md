---
title: Interaction Fingerprint
tags:
  - sbdd
  - molecular-modeling
  - protein-ligand
---

# Interaction Fingerprint

An interaction fingerprint encodes protein-ligand contacts or interaction types as a vector. It is useful for comparing poses, clustering docking results, summarizing binding modes, and building interpretable features.

For residue or atom interaction indicators:

$$
z_k
=
\mathbf{1}[\text{interaction } k \text{ is present}]
$$

The fingerprint is:

$$
z = (z_1,\ldots,z_K)
$$

Similarity between two interaction fingerprints can be computed with overlap or Tanimoto-style measures:

$$
T(z,z')
=
\frac{z\cdot z'}
{\lVert z\rVert_1+\lVert z'\rVert_1-z\cdot z'}
$$

## Key Ideas

- Interaction fingerprints summarize contacts, not full geometry.
- The feature definition depends on distance cutoffs, atom typing, residue selection, and interaction rules.
- Similar fingerprints can hide different poses if geometry is coarse.
- They are useful diagnostic features but should not replace pose quality checks.

## Feature Contract

An interaction fingerprint should state:

| Field | Question |
| --- | --- |
| receptor unit | residue, atom, pharmacophore point, pocket region, or interaction site |
| ligand unit | atom, fragment, pharmacophore feature, or whole-ligand contact |
| interaction types | hydrophobic, H-bond, ionic, aromatic, metal, water-mediated, clash |
| distance/angle rules | cutoffs, angular constraints, donor/acceptor typing |
| chemical state | hydrogens, protonation, tautomer, charge, metals, cofactors |
| aggregation | binary, count, weighted, residue-level, atom-level, or graph feature |

Changing any of these fields changes the feature space:

$$
z
=
\operatorname{IFP}(P,L,\hat{X};\rho)
$$

where $\rho$ is the rule set.

## Comparison Boundary

The same fingerprint can be used for different claims:

| Use | Metric | Risk |
| --- | --- | --- |
| compare poses for one complex | Tanimoto or overlap | coarse contacts may hide wrong geometry |
| cluster docking poses | fingerprint distance | clusters depend on interaction rules |
| screen molecules | similarity to reference interaction pattern | reference pose or ligand can leak target knowledge |
| model input | binary/count features | hand-crafted features may dominate learned model |
| interpret prediction | highlighted contacts | contact does not prove causal binding contribution |

An interaction fingerprint similarity is not a replacement for pose RMSD, clash checks, affinity, or enrichment.

## Leakage Risks

- Using a known native ligand interaction pattern to filter test poses can leak the answer.
- Residue selection based on the bound ligand can make the task easier than deployment.
- Protonation or hydrogen placement rules can create inconsistent H-bond features across splits.
- Comparing across targets without residue mapping can produce misleading similarity.

## Practical Checks

- Which interaction types are encoded?
- Are cutoffs and atom typing rules documented?
- Are waters, metals, cofactors, and protonation states included or ignored?
- Does the fingerprint compare poses within one target or across targets?
- Is it used for interpretation, clustering, filtering, or model input?
- Is the reference interaction pattern available at inference time?
- Are geometry checks reported alongside fingerprint similarity?

## Related

- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/pose-rmsd|Pose RMSD]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/protein-modeling/binding-site|Binding site]]
