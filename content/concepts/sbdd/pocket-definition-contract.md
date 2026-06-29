---
title: Pocket Definition Contract
tags:
  - sbdd
  - pocket
  - leakage
---

# Pocket Definition Contract

A pocket definition contract states how the binding site is chosen before modeling or evaluation. Pocket definition is part of the task, not a harmless preprocessing detail, because it controls what information is available to the model.

$$
B = \phi(P, C)
$$

where $P$ is the protein structure and $C$ is the allowed context for selecting the pocket.

## Pocket Modes

| Mode | Allowed context | Use for | Risk |
| --- | --- | --- | --- |
| known site | public annotation or prior biological knowledge | focused screening, known target workflow | annotation may be benchmark-specific |
| predicted site | pocket predictor output from receptor only | prospective or blind-ish workflow | predictor quality becomes part of method |
| ligand-defined | residues near a bound ligand | retrospective pose analysis | test ligand pose leakage |
| blind grid | whole protein or broad search region | blind docking | larger search space and lower baseline |
| template-derived | homolog or template complex | structure transfer | template and homolog leakage |
| user-defined box | manually specified center/radius | operational docking workflow | may hide unavailable expert knowledge |

## Required Fields

| Field | Record |
| --- | --- |
| receptor source | experimental, predicted, template, ensemble, prepared structure |
| pocket source | annotation, predictor, ligand, template, manual box, blind grid |
| selection rule | residue cutoff, grid size, center, chain, cofactors, exclusions |
| inference availability | whether the same pocket rule is available without the test ligand |
| split interaction | whether pocket source leaks homolog, template, ligand, or assay information |
| evaluation impact | how failures in pocket definition are counted |

## Claim Boundary

| Claim | Pocket requirement |
| --- | --- |
| pose recovery around known ligand | ligand-defined pocket can be acceptable if stated explicitly |
| prospective docking | pocket must be known, predicted, or defined without test pose |
| virtual screening | pocket rule must be fixed before candidate labels are used |
| structure-aware generation | conditioning pocket must match deployment information |
| new target generalization | pocket source must respect protein-family/template split |

## Checks

- Is the pocket chosen before seeing the test ligand pose?
- Would the same pocket be available for a novel ligand or target?
- Are failed pocket predictions counted or silently filtered?
- Does the pocket rule use benchmark-only annotations?
- Is the receptor structure source independent of the evaluated complex?

## Related

- [[entities/pocket|Pocket]]
- [[molecular-modeling/structure-based/index|Structure-Based Modeling]]
- [[concepts/sbdd/docking-workflow|Docking workflow]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[concepts/sbdd/template-leakage|Template leakage]]
