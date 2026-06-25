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

With a margin, the pairwise hinge loss is:

$$
\mathcal{L}_{\mathrm{hinge}}
=
\max(0, \gamma - s_i + s_j)
$$

where $\gamma>0$ is the desired score margin.

For listwise ranking, a probability over candidates can be formed with softmax:

$$
P(i\mid q)
=
\frac{\exp(s_i)}
{\sum_{j=1}^{N}\exp(s_j)}
$$

This is useful when the model should assign probability mass to the best candidates in a set.

## Evaluation Focus

Ranking quality depends on where mistakes occur. A poor ordering near the bottom of a list may not matter for top-k retrieval, while a single false high-ranked candidate can matter for screening.

## Checks

- Is the goal top-k retrieval, full ordering, pairwise preference, or calibrated probability?
- Are positives and negatives comparable under the same sampling process?
- Does the metric emphasize early ranking, such as top-k, enrichment, or average precision?
- Are ties, duplicates, and near-duplicates handled?
- Is the candidate set at evaluation time drawn from the same process as deployment?

## Related

- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[concepts/learning/preference-optimization|Preference optimization]]
- [[concepts/machine-learning/loss-function|Loss function]]
