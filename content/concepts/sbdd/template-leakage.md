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

## Checks

- Are train/test proteins separated by family or sequence identity?
- Are ligands separated by scaffold or similarity?
- Was any bound ligand, close analog, or protein-ligand complex used as a template?
- Are template databases and cutoffs documented?
- Are pose quality, affinity, and enrichment claims evaluated under the same leakage policy?

## Related

- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/sbdd/docking-workflow|Docking workflow]]
- [[concepts/sbdd/pose-quality|Pose quality]]
