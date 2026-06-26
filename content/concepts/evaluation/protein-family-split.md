---
title: Protein Family Split
tags:
  - evaluation
  - methodology
  - proteins
---

# Protein Family Split

A protein family split partitions data by sequence or structural homology so that related proteins do not appear in both train and test. Random splits over protein data leak through homology, letting a model memorize a family rather than learn transferable signal.

The grouped split constraint is:

$$
g(p_i)=g(p_j)
\Rightarrow
s(p_i)=s(p_j)
$$

Here $g$ maps a protein to a sequence or structure family and $s$ maps it to train, validation, or test.

The split is usually implemented by clustering:

$$
p_i \sim p_j
\quad
\text{if}
\quad
\operatorname{identity}(p_i,p_j)\ge \tau
$$

then assigning whole clusters to splits. Smaller $\tau$ makes the test set more remote from training proteins.

## Practical Checks

- Cluster by sequence identity (e.g. via MMseqs2/CD-HIT) and split whole clusters.
- Choose an identity threshold matched to the generalization claim being made.
- Consider structural similarity when sequences diverge but folds are shared.
- Keep entire clusters on one side — partial overlap reintroduces leakage.
- State the clustering method and threshold in every reported result.

## Threshold Interpretation

| Split rule | Typical claim | Remaining risk |
|---|---|---|
| random rows | interpolation over seen families | strong homolog leakage |
| high-identity clusters | near-duplicate control | family-level memorization may remain |
| low-identity clusters | remote homolog transfer | shared folds or domains may remain |
| fold/domain split | structural novelty | limited data and label shift |
| external target set | deployment-like transfer | source and assay shift |

The right threshold depends on the claim. A lenient threshold can be fine for interpolation, but weak for new-family claims.

## What It Proves

A protein-family split tests whether a model generalizes beyond close homologs:

$$
\operatorname{family}(p_i)=\operatorname{family}(p_j)
\Rightarrow
s(p_i)=s(p_j)
$$

It does not automatically prove generalization to new folds, new assays, new ligands, or new experimental sources. Those require separate split rules or external test sets. For protein-ligand tasks, it should be interpreted together with [[concepts/sbdd/protein-ligand-split|Protein-ligand split]] so target novelty is not confused with ligand-side interpolation.

## Reporting Fields

| Field | Why it matters |
|---|---|
| clustering tool and version | identity definitions differ |
| identity threshold $\tau$ | controls novelty strength |
| representative sequence policy | affects cluster membership |
| train/validation/test cluster counts | reveals imbalance |
| label distribution per split | detects target and assay shift |
| overlap audit | proves grouped splitting was enforced |

For structure-based tasks, also report whether proteins share pockets, ligands, domains, or structures across splits.

## Related

- [[concepts/evaluation/evaluation-protocol|Evaluation protocol]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/sbdd/protein-ligand-split|Protein-ligand split]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/index|Evaluation]]
