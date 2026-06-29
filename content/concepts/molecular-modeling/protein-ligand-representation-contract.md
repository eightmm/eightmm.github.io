---
title: Protein-ligand Representation Contract
tags:
  - molecular-modeling
  - representation
  - protein-ligand
---

# Protein-ligand Representation Contract

A protein-ligand representation contract records how protein, ligand, pocket, pose, and interaction evidence become model input. It prevents a model note from mixing object identity, geometry, assay labels, and evaluation-only information.

$$
r(u)
=
r(P, L, B, X, c)
$$

where $P$ is the protein or target, $L$ is the ligand, $B$ is the pocket or binding site rule, $X$ is coordinate evidence, and $c$ is task context such as assay or structure source.

## Required Fields

| Field | Record |
| --- | --- |
| Protein unit | sequence, chain, isoform, construct, mutation, species, structure source |
| Ligand unit | molecule identity, salt, stereo, tautomer, protonation, charge, conformer |
| Pocket rule | known site, predicted site, ligand-defined site, blind grid, residue cutoff |
| Coordinate source | experimental, predicted, docked, minimized, generated, simulated |
| Atom/residue mapping | how nodes correspond across reference, generated, and evaluated structures |
| Pair context | target, assay, endpoint, unit, censoring, source, complex identity |
| Availability | which features are available at inference time |
| Split boundary | scaffold, protein family, complex pair, assay/source, template, time |

## Representation Patterns

| Pattern | Input | Main risk |
| --- | --- | --- |
| ligand-only | ligand graph, fingerprint, conformer | target context hidden or ignored |
| protein-only | sequence or structure embedding | ligand-specific claim unsupported |
| late fusion | separate protein and ligand embeddings | interaction geometry may be weak |
| pocket-conditioned | ligand plus pocket representation | pocket may use unavailable ligand information |
| complex graph | protein-ligand graph with edges | edge construction may encode the answer |
| coordinate model | atom/residue coordinates and updates | frame, chirality, alignment, coordinate-source leakage |

## Leakage Checks

- Does the pocket require the ground-truth ligand pose?
- Does the graph contain edges computed from the reference complex when the claim is blind prediction?
- Are close ligand analogs or homologous proteins split across train and test?
- Are conformers generated using a protocol that sees evaluation labels or reference poses?
- Are assay labels collapsed across incompatible endpoints, units, or sources?
- Is the representation available under the intended deployment setting?

## Related

- [[molecular-modeling/representation-routes|Representation Routes]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]
- [[concepts/sbdd/pocket-definition-contract|Pocket definition contract]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[concepts/evaluation/leakage|Leakage]]
