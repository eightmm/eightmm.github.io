---
title: Sequence Identity Clustering
tags:
  - protein-modeling
  - evaluation
  - data
---

# Sequence Identity Clustering

Sequence identity clustering groups protein sequences so that highly similar sequences stay on the same side of a split. It is a core guardrail against leakage in protein modeling.

For two aligned sequences $s_i$ and $s_j$:

$$
\operatorname{identity}(s_i, s_j)
= \frac{1}{L}\sum_{\ell=1}^{L}
\mathbf{1}[s_{i,\ell}=s_{j,\ell}]
$$

$L$ is aligned length and $\mathbf{1}$ is an indicator function. A clustering threshold $\tau$ defines which sequences are too similar to separate:

$$
\operatorname{identity}(s_i, s_j) \ge \tau
$$

## Why It Matters

Random protein splits often place homologous or near-duplicate proteins in both train and test. The model then reports memorization or family lookup instead of generalization.

## Checks

- What identity threshold is used?
- Are chains, isoforms, fragments, and duplicate records normalized before clustering?
- Are all members of a cluster assigned to the same split?
- Does the benchmark need protein-family, target-family, or structure-level separation?
- Is the threshold public and reproducible?

## Related

- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]]
- [[concepts/protein-modeling/sequence-structure-alignment|Sequence-structure alignment]]
- [[concepts/evaluation/leakage|Leakage]]
