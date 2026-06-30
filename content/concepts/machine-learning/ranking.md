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

Only the ordering is required for ranking:

$$
s_i > s_j
\Rightarrow
x_i \succ x_j
$$

The score scale may be arbitrary unless it is explicitly calibrated as a probability, utility, affinity, or reward.

## Pointwise, Pairwise, Listwise

Ranking objectives differ by what training example they use.

| Objective Family | Unit | Example |
| --- | --- | --- |
| pointwise | candidate with label | predict relevance, activity, click, preference score |
| pairwise | ordered pair | candidate $i$ should outrank candidate $j$ |
| listwise | candidate set | assign mass or order over a whole list |

Pointwise ranking often reuses classification or regression losses, but the evaluation claim is still about ordering in a candidate pool.

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

Pair construction is part of the data contract:

$$
\mathcal{P}
=
\{(i,j): x_i \succ x_j\}
$$

If negatives are sampled too easily, a model can look good without learning the hard ordering needed at deployment.

For listwise ranking, a probability over candidates can be formed with softmax:

$$
P(i\mid q)
=
\frac{\exp(s_i)}
{\sum_{j=1}^{N}\exp(s_j)}
$$

This is useful when the model should assign probability mass to the best candidates in a set.

The listwise cross-entropy for a target distribution $r_i$ over the list is:

$$
\mathcal{L}_{\mathrm{list}}
=
-\sum_{i=1}^{N} r_i \log P(i\mid q)
$$

where $r_i$ may be one-hot, graded relevance normalized into a distribution, or a preference-derived target.

## Evaluation Focus

Ranking quality depends on where mistakes occur. A poor ordering near the bottom of a list may not matter for top-k retrieval, while a single false high-ranked candidate can matter for screening.

For a ranked list, precision at $k$ is:

$$
\mathrm{Precision@}k
=
\frac{1}{k}
\sum_{i=1}^{k}
\mathbb{1}[\text{item}_i\ \text{is relevant}]
$$

Recall at $k$ is:

$$
\mathrm{Recall@}k
=
\frac{
\sum_{i=1}^{k}\mathbb{1}[\text{item}_i\ \text{is relevant}]
}{
\#\text{relevant items in candidate set}
}
$$

These metrics depend on the candidate set denominator. Changing the candidate pool can change the metric without changing the model.

## Candidate Pool Contract

| Field | Why |
| --- | --- |
| query unit | a ranking is always relative to a query or context |
| candidate source | defines the denominator and difficulty |
| positive definition | relevance, hit, preference, active, accepted answer |
| negative definition | random, hard negative, decoy, unjudged item |
| tie handling | affects metric and pair construction |
| duplicate policy | prevents near-duplicate inflation |
| split unit | prevents same query, scaffold, target, user, or document leakage |

## Ranking vs Classification vs Regression

| If the goal is | Prefer |
| --- | --- |
| probability of a class | [[concepts/machine-learning/classification|Classification]] |
| accurate numeric value | [[concepts/machine-learning/regression|Regression]] |
| top-k ordering or screening | this page |
| calibrated expected utility | decision rule plus calibration/evaluation |

Regression scores are often used for ranking, but a low RMSE does not guarantee good early enrichment. Classification probabilities are often used for ranking, but good AUROC does not guarantee the top of the list is useful.

## Checks

- Is the goal top-k retrieval, full ordering, pairwise preference, or calibrated probability?
- Are positives and negatives comparable under the same sampling process?
- Does the metric emphasize early ranking, such as top-k, enrichment, or average precision?
- Are ties, duplicates, and near-duplicates handled?
- Is the candidate set at evaluation time drawn from the same process as deployment?
- What is the denominator of the ranked candidate pool?
- Are hard negatives included, or only easy random negatives?
- Is the score interpreted only ordinally, or also as calibrated probability/utility?

## Related

- [[concepts/machine-learning/classification|Classification]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/machine-learning/decision-rule|Decision rule]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/precision-recall|Precision and recall]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/tasks/reranking|Reranking]]
- [[concepts/tasks/similarity-search|Similarity search]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[concepts/learning/preference-optimization|Preference optimization]]
- [[concepts/machine-learning/loss-function|Loss function]]
