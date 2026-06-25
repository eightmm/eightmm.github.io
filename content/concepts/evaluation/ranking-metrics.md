---
title: Ranking Metrics
tags:
  - evaluation
  - ranking
  - metrics
---

# Ranking Metrics

Ranking metrics evaluate whether high-quality items appear near the top of a ranked list. They are central to retrieval, virtual screening, recommendation, reranking, and agent candidate selection.

For a ranked list with binary relevance labels $\mathrm{rel}_k\in\{0,1\}$, precision at $K$ is:

$$
P@K
=
\frac{1}{K}
\sum_{k=1}^{K}
\mathrm{rel}_k
$$

Recall at $K$ is:

$$
R@K
=
\frac{\sum_{k=1}^{K}\mathrm{rel}_k}
{\sum_{k=1}^{N}\mathrm{rel}_k}
$$

Discounted cumulative gain emphasizes early ranks:

$$
\mathrm{DCG}@K
=
\sum_{k=1}^{K}
\frac{2^{\mathrm{rel}_k}-1}{\log_2(k+1)}
$$

Normalized DCG divides by the best possible DCG:

$$
\mathrm{NDCG}@K
=
\frac{\mathrm{DCG}@K}{\mathrm{IDCG}@K}
$$

## Scientific Screening Metrics

Virtual screening often uses enrichment-style metrics. If $A_K$ active items appear in the top $K$, total actives are $A$, and total items are $N$:

$$
\mathrm{EF}@K
=
\frac{A_K/K}{A/N}
$$

This asks how much better the top-ranked subset is than random selection.

## Practical Checks

- Is the decision based on the top few items or the whole ranking?
- Are relevance labels complete, partial, noisy, or assay-dependent?
- Does the benchmark contain many near-duplicates that make ranking easier?
- For virtual screening, are decoys and negatives realistic for the target setting?

## Related

- [[concepts/machine-learning/ranking|Ranking]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[concepts/evaluation/negative-set|Negative set]]
