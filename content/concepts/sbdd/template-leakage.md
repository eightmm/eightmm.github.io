---
title: Template Leakage
tags:
  - sbdd
  - evaluation
  - leakage
---

# Template Leakage

Template leakage happens when evaluation examples are indirectly seen through homologous structures, bound complexes, close ligands, or template databases used during model construction.

For a test complex $(P_{\mathrm{test}}, L_{\mathrm{test}})$, leakage risk increases when training or template data contains:

$$
\operatorname{sim}(P_{\mathrm{test}}, P_{\mathrm{train}}) \ge \tau_P
\quad \text{and} \quad
\operatorname{sim}(L_{\mathrm{test}}, L_{\mathrm{train}}) \ge \tau_L
$$

$\tau_P$ is a protein similarity threshold and $\tau_L$ is a ligand similarity threshold. The exact thresholds depend on the benchmark policy.

## Why It Matters

Structure-based models can look strong when the test target, binding mode, or ligand chemotype is already represented in training data or template databases. This is especially risky for docking, pose prediction, protein structure prediction, and protein-ligand interaction models.

## Leakage Sources

| Source | What Leaks | Typical Symptom |
| --- | --- | --- |
| protein templates | fold, pocket geometry, side-chain arrangement | strong pose results on homologous targets |
| bound complexes | binding mode and pocket-ligand geometry | near-native poses without real generalization |
| ligand analogs | scaffold, substructure, interaction pattern | inflated enrichment or affinity prediction |
| benchmark curation | duplicate complexes or same assay source | random split much stronger than family/scaffold split |
| preprocessing databases | templates used during structure prediction, MSA, or pocket definition | deployment cannot reproduce the same context |
| generated decoys | negatives easier than real inactive molecules | high AUROC but poor prospective screening |

## Split Contract

For a structure-based benchmark, state which axes are held out:

$$
g(x_i)
=
\left(
g_{\mathrm{protein}}(P_i),
g_{\mathrm{ligand}}(L_i),
g_{\mathrm{assay}}(a_i),
g_{\mathrm{time}}(i)
\right)
$$

The claim depends on the held-out axes:

| Held-Out Axis | Supported Claim |
| --- | --- |
| ligand scaffold | new chemotypes for known or related targets |
| protein family | new target families for known or related chemistry |
| protein-ligand pair | new combinations, not necessarily new entities |
| assay/source | robustness across measurement protocols |
| time | future-like deployment from past data |

If only one axis is held out, do not claim generalization on all axes.

## Template Audit

Before trusting a benchmark claim, ask whether any stage used deployment-unavailable templates:

- protein structure prediction;
- pocket cropping or binding-site definition;
- receptor alignment or side-chain preparation;
- ligand pose initialization;
- database search, MSA, or similarity retrieval;
- post hoc filtering based on known complex geometry.

## Checks

- Are train/test proteins separated by family or sequence identity?
- Are ligands separated by scaffold or similarity?
- Is the combined [[concepts/sbdd/protein-ligand-split|protein-ligand split]] strict enough for the benchmark claim?
- Was any bound ligand, close analog, or protein-ligand complex used as a template?
- Are template databases and cutoffs documented?
- Are pose quality, affinity, and enrichment claims evaluated under the same leakage policy?
- Are sequence, structure, ligand, and assay leakage checked together rather than one at a time?
- Does the preprocessing pipeline use information available at deployment?

## Related

- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[concepts/sbdd/docking-workflow|Docking workflow]]
- [[concepts/sbdd/pose-quality|Pose quality]]
