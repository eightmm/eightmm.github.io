---
title: Ranking
tags:
  - machine-learning
  - evaluation
---

# Ranking

Ranking orders candidates by relevance, utility, probability, affinity, score, or preference. It is central to search, recommendation, retrieval, virtual screening, reranking, and model selection.

A ranking model assigns each candidate $x_i$ a score:

$$
s_i = f_\theta(q, x_i)
$$

where $q$ is an optional query or context. Candidates are sorted by $s_i$.

A pairwise ranking objective can be written as:

$$
\mathcal{L}_{\mathrm{rank}}
= -\log \sigma(s_i-s_j)
$$

where candidate $i$ should be ranked above candidate $j$.

## Checks

- Is the goal top-k retrieval, full ordering, pairwise preference, or calibrated probability?
- Are positives and negatives comparable under the same sampling process?
- Does the metric emphasize early ranking, such as top-k, enrichment, or average precision?
- Are ties, duplicates, and near-duplicates handled?

## Related

- [[concepts/evaluation/metric|Metric]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[concepts/learning/preference-optimization|Preference optimization]]
- [[concepts/machine-learning/loss-function|Loss function]]
