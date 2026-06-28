---
title: Molecular and Ligand Modeling
aliases:
  - computational-biology/molecular-ligand
  - computational-biology/molecules
  - bio/molecular-ligand
tags:
  - computational-biology
  - molecular-modeling
---

# Molecular and Ligand Modeling

Molecular and ligand modeling covers small-molecule identity, representation, standardization, conformers, property prediction, retrieval, and generation. A ligand is not a different object type from a molecule; it is a molecule in a binding or target context.

$$
r_L
=
\phi(L, c)
$$

where $L$ is a standardized molecular object and $c$ can include pH, protonation, tautomer policy, conformer source, assay context, target, or pocket.

## Route Map

| Question | Start | Watch |
| --- | --- | --- |
| What chemical object is modeled? | [Molecules](/molecular-modeling/molecules), [Molecule](/entities/molecule), [Ligand](/entities/ligand) | salts, mixtures, stereochemistry, tautomer, protonation |
| What representation does the model see? | [RDKit](/concepts/molecular-modeling/rdkit), [Molecular graph](/concepts/molecular-modeling/molecular-graph), [SMILES](/concepts/molecular-modeling/smiles), [Molecular fingerprint](/concepts/molecular-modeling/molecular-fingerprint) | equivalent molecules can become different inputs |
| Is 3D conformation involved? | [Conformer](/concepts/molecular-modeling/conformer), [Force field](/concepts/molecular-modeling/force-field) | conformer source and minimization protocol can dominate |
| Is the molecule target-conditioned? | [Interaction Modeling](/molecular-modeling/interactions), [Target-assay-label contract](/entities/target-assay-label) | molecule-only claims do not cover target-specific activity |
| Is structure or docking central? | [Structure-Based Modeling](/molecular-modeling/structure-based) | pose, pocket, scoring, and geometry require separate checks |

## Main Subroutes

| Area | Use For | Start |
| --- | --- | --- |
| Chemical identity | standardization, salt stripping, stereo, tautomer, protonation | [Molecules](/molecular-modeling/molecules) |
| 2D representation | SMILES, molecular graph, fingerprint, scaffold, similarity | [RDKit](/concepts/molecular-modeling/rdkit), [Molecular graph](/concepts/molecular-modeling/molecular-graph) |
| 3D representation | conformer, force field, shape, geometry, docking input | [Conformer](/concepts/molecular-modeling/conformer) |
| Property and retrieval | property prediction, similarity search, candidate ranking | [Molecular property prediction](/concepts/molecular-modeling/molecular-property-prediction) |
| Generation | valid molecule samples, constrained generation, scaffold editing | [Molecular generation](/concepts/generative-models/molecular-generation) |

## Chemical State Checklist

| State | Ask |
| --- | --- |
| Salt / mixture | is the modeled molecule the parent, salt, mixture, or assay record? |
| Stereochemistry | are unspecified and specified stereocenters handled consistently? |
| Tautomer | is one canonical tautomer chosen, or are several states possible? |
| Protonation / charge | is the state compatible with pH, assay, docking, or force field assumptions? |
| Conformer | is 3D geometry generated, experimental, minimized, or reused from a complex? |
| Deduplication | are equivalent molecules collapsed before split construction? |

## RDKit Boundary

RDKit is usually the implementation layer for molecule parsing, standardization, fingerprints, descriptors, substructure search, and conformers. Treat these settings as method choices, not invisible preprocessing.

| RDKit Use | Record |
| --- | --- |
| canonical SMILES | standardization and stereo policy |
| Morgan fingerprint | radius, bit length, count/binary mode, chirality flag |
| descriptor vector | descriptor list and missing-value policy |
| scaffold split | standardized molecule used for scaffold extraction |
| conformer generation | seed, conformer count, force field, minimization and failure policy |

## Boundary

Use this page for molecule-only or ligand-preparation questions. Use [[molecular-modeling/interactions|Interaction modeling]] when the row is a molecule-target-assay relation. Use [[molecular-modeling/structure-based/index|Structure-based modeling]] when a pocket, pose, coordinate frame, or protein-ligand complex is part of the claim.

## Checks

- Was molecular standardization done before deduplication and splitting?
- Are stereochemistry, charge, protonation, tautomer, and conformer policy explicit?
- Is the split scaffold-based, temporal, target-aware, or assay-aware?
- Does the metric test property prediction, retrieval, generation, docking, or screening?

## Related

- [[molecular-modeling/index|Computational Biology]]
- [[molecular-modeling/entities|Objects and entities]]
- [[molecular-modeling/molecules|Molecules]]
- [[molecular-modeling/interactions|Interaction modeling]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
