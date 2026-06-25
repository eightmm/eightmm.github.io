---
title: Scaffold Split
tags:
  - evaluation
  - methodology
  - cheminformatics
---

# Scaffold Split

A scaffold split groups molecules by their core structure (e.g. Bemis–Murcko scaffold) and keeps each scaffold entirely in one split. It estimates how a model generalizes to chemotypes it has not seen, unlike a random split that scatters close analogs across train and test.

The grouped split constraint is:

$$
g(m_i)=g(m_j)
\Rightarrow
s(m_i)=s(m_j)
$$

Here $g$ maps a molecule to its scaffold and $s$ maps it to train, validation, or test.

## Practical Checks

- Compute scaffolds consistently and assign whole scaffold groups to a single split.
- Expect lower, more honest scores than random splits — that is the point.
- Watch for tiny test scaffolds that make metrics noisy.
- Combine with duplicate/near-duplicate removal to avoid hidden leakage.
- Report the split method explicitly; "test accuracy" is meaningless without it.

## What It Proves

A scaffold split does not prove broad chemical generalization by itself. It tests a specific shift:

$$
p_{\mathrm{test}}(m)
\ne
p_{\mathrm{train}}(m)
\quad
\text{through scaffold grouping}
$$

The result still depends on assay quality, target context, class balance, scaffold group size, and whether molecules were standardized consistently. For target-conditioned molecular tasks, scaffold split should be interpreted together with [[concepts/sbdd/protein-ligand-split|Protein-ligand split]] so ligand-side novelty is not confused with target-side generalization.

## Related

- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[concepts/evaluation/index|Evaluation]]
