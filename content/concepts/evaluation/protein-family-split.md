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

## Practical Checks

- Cluster by sequence identity (e.g. via MMseqs2/CD-HIT) and split whole clusters.
- Choose an identity threshold matched to the generalization claim being made.
- Consider structural similarity when sequences diverge but folds are shared.
- Keep entire clusters on one side — partial overlap reintroduces leakage.
- State the clustering method and threshold in every reported result.

## Related

- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/index|Evaluation]]
