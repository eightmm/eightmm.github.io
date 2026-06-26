---
title: Receptor and Ligand Preparation
tags:
  - sbdd
  - docking
  - data-preprocessing
---

# Receptor and Ligand Preparation

Receptor and ligand preparation converts raw structures and molecules into model-ready inputs for docking, scoring, pose evaluation, or property prediction. Preparation choices can dominate results, so they should be treated as part of the method.

A preparation pipeline can be written as:

$$
(P_{\mathrm{raw}}, L_{\mathrm{raw}})
\xrightarrow{\phi_{\mathrm{prep}}}
(P_{\mathrm{model}}, L_{\mathrm{model}})
$$

where $\phi_{\mathrm{prep}}$ includes cleaning, standardization, protonation, conformer handling, and binding-site definition.

Preparation is not neutral:

$$
\hat{y}
=
f_\theta(\phi_{\mathrm{prep}}(P_{\mathrm{raw}},L_{\mathrm{raw}}))
$$

Changing $\phi_{\mathrm{prep}}$ can change the model input, docking pose, score, and failure rate even when the model weights stay fixed.

## Receptor Preparation

- Choose chain, biological assembly, and relevant structure.
- Handle missing residues, alternate conformations, waters, ions, metals, cofactors, and ligands.
- Assign protonation or charge states when needed.
- Define the pocket, docking box, or local context.
- Decide whether the receptor is rigid, flexible, or ensemble-based.

| Decision | Common options | Risk |
|---|---|---|
| chain/assembly | asymmetric unit, biological assembly, selected chain | wrong interface or missing binding context |
| waters | remove all, keep structural waters, model explicitly | water-mediated contacts disappear |
| cofactors/metals | remove, keep, parameterize | active-site chemistry changes |
| missing residues | ignore, model, crop | pocket geometry becomes inconsistent |
| apo/holo structure | apo, holo, predicted | conformation may leak ligand information or miss induced fit |

## Ligand Preparation

- Define [[concepts/molecular-modeling/molecular-identity|molecular identity]] before deduplication and split assignment.
- Standardize salt, charge, aromaticity, tautomer, and stereochemistry policy.
- Enumerate or choose protonation and tautomer states when relevant.
- Generate conformers if the workflow requires 3D inputs.
- Preserve identifiers without using them as model features.
- Track input hash and featurizer/preparation version.

Ligand state enumeration can be represented as:

$$
\mathcal{S}(L)
=
\{L^{(1)},\ldots,L^{(m)}\}
$$

where each state may differ by tautomer, protonation, stereochemistry, salt handling, or conformer. A benchmark should state whether it evaluates the best enumerated state, a selected canonical state, or all states.

## Failure Accounting

Failed preparation is part of the result:

$$
\operatorname{failure\ rate}
=
\frac{n_{\mathrm{failed}}}{n_{\mathrm{total}}}
$$

If failures are silently removed, the reported metric may only describe easy molecules or clean receptors. This is especially risky when failures correlate with charge, size, metal coordination, flexible ligands, or low-quality structures.

## Reproducibility Contract

| Field | Why it matters |
|---|---|
| input structure ID and chain policy | defines the receptor object |
| preparation software and version | changes charges, hydrogens, atom typing |
| pH/protonation assumptions | changes interaction geometry |
| tautomer/stereo policy | changes ligand identity |
| conformer generation settings | changes search starting points |
| pocket or box definition | changes docking search space |
| failure handling | defines denominator of metrics |

## Checks

- Are receptor and ligand preparation versions recorded?
- Is pocket definition independent from test-set information?
- Are stereochemistry and protonation rules consistent across splits?
- Are failed preparations counted and reported?
- Are preparation failures correlated with target labels or molecular classes?
- Are waters, metals, cofactors, and alternate conformations handled by a stated rule?
- Is a ligand-defined pocket allowed by the deployment scenario?
- Are raw and prepared molecule identifiers traceable without exposing private paths?

## Related

- [[concepts/sbdd/docking-workflow|Docking workflow]]
- [[concepts/protein-modeling/binding-site|Binding site]]
- [[concepts/molecular-modeling/molecular-identity|Molecular identity]]
- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]
- [[concepts/systems/reproducibility|Reproducibility]]
