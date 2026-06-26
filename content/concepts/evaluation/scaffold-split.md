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

For a scaffold set $\mathcal{G}$, a split assigns entire groups:

$$
\mathcal{G}
=
\mathcal{G}_{\mathrm{train}}
\dot{\cup}
\mathcal{G}_{\mathrm{val}}
\dot{\cup}
\mathcal{G}_{\mathrm{test}}
$$

and each molecule follows its scaffold group:

$$
s(m)=s(g(m))
$$

## Practical Checks

- Compute scaffolds consistently and assign whole scaffold groups to a single split.
- Expect lower, more honest scores than random splits — that is the point.
- Watch for tiny test scaffolds that make metrics noisy.
- Combine with duplicate/near-duplicate removal to avoid hidden leakage.
- Report the split method explicitly; "test accuracy" is meaningless without it.

## What Counts as a Scaffold

| Choice | Meaning | Risk |
|---|---|---|
| Bemis-Murcko scaffold | ring systems and linkers | common default, but may over-group or under-group |
| generic scaffold | atom/bond types generalized | stronger chemotype holdout, less specific |
| scaffold network | hierarchy of cores | split choice becomes a modeling decision |
| similarity cluster | fingerprint or graph similarity cluster | threshold controls novelty strength |

The split is only as meaningful as the scaffold definition. Two papers can both say "scaffold split" while testing different shifts.

## Split Axis Selection

| Claim | Better split axis |
|---|---|
| new chemotypes for same assay family | scaffold or similarity cluster split |
| new protein targets | [[concepts/evaluation/protein-family-split|Protein family split]] |
| new target-ligand pairs | [[concepts/sbdd/protein-ligand-split|Protein-ligand split]] |
| new assay source | assay/source split |
| deployment over time | temporal split |

For protein-ligand data, a ligand scaffold split alone does not prove target generalization.

## Metric Effects

Scaffold split often changes class balance and target distribution:

$$
p_{\mathrm{test}}(y\mid g)
\ne
p_{\mathrm{train}}(y\mid g)
$$

Report both example counts and scaffold counts:

| Split | Examples | Scaffolds | Positives | Negatives |
|---|---:|---:|---:|---:|
| train | `to report` | `to report` | `to report` | `to report` |
| validation | `to report` | `to report` | `to report` | `to report` |
| test | `to report` | `to report` | `to report` | `to report` |

Otherwise a metric drop may reflect harder chemistry, class imbalance, smaller test groups, or label-source shift.

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
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[concepts/evaluation/index|Evaluation]]
