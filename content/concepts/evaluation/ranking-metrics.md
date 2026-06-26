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

## Candidate Pool Contract

Ranking metrics are conditional on the candidate set:

$$
M(q)
=
M(\operatorname{rank}_\theta(\mathcal{C}_q),\ \mathrm{rel}_q)
$$

where $q$ is a query, $\mathcal{C}_q$ is the candidate pool, and $\mathrm{rel}_q$ gives relevance labels. A score from one candidate pool does not automatically transfer to another pool.

| Pool Type | Metric Risk |
| --- | --- |
| curated negatives | ranking may be easier than deployment |
| random decoys | artificial property biases can inflate enrichment |
| incomplete labels | unobserved positives are counted as negatives |
| near-duplicate candidates | scaffold or sequence leakage can dominate top ranks |
| per-query candidate sets | macro and micro averages answer different questions |

For molecular screening, state whether negatives are measured inactive compounds, presumed negatives, decoys, generated candidates, or unlabeled compounds.

## Aggregation

For multiple queries:

$$
\operatorname{macro}\text{-}M
=
\frac{1}{Q}
\sum_{q=1}^{Q}M(q)
$$

Macro averaging weights each query equally. Micro-style aggregation pools candidates and can be dominated by large or easy queries. For protein targets, assay panels, retrieval questions, or agent candidate lists, state the aggregation unit.

## Cutoff Choice

$K$ should match the downstream action. For screening, top $1\%$, top $K$, or budget-limited selection can be more meaningful than full-list metrics. Choose $K$ on validation or from a fixed application budget, not after looking at the test ranking.

## Practical Checks

- Is the decision based on the top few items or the whole ranking?
- Are relevance labels complete, partial, noisy, or assay-dependent?
- Does the benchmark contain many near-duplicates that make ranking easier?
- For virtual screening, are decoys and negatives realistic for the target setting?
- Is the candidate pool fixed, query-specific, generated, or deployment-like?
- Are metrics macro-averaged by query/target or micro-averaged over all candidates?

## Related

- [[concepts/machine-learning/ranking|Ranking]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[concepts/evaluation/negative-set|Negative set]]
