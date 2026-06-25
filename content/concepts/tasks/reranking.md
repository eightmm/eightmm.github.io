---
title: Reranking
tags:
  - tasks
  - ranking
  - retrieval
---

# Reranking

Reranking reorders a candidate set produced by an earlier retrieval, search, or generation stage. It is useful when a fast first stage has high recall but weak precision, and a slower model can evaluate fewer candidates more carefully.

For a query $q$, first-stage retrieval gives:

$$
C_k(q)
=
\operatorname{TopK}_{x \in \mathcal{D}} s_0(q,x)
$$

A reranker assigns a second score:

$$
s_1(q,x) = h_\psi(q,x,\eta_x)
$$

and returns:

$$
\operatorname{Rerank}(q)
=
\operatorname{sort}_{x \in C_k(q)} s_1(q,x)
$$

where $s_0$ is the cheap retrieval score, $h_\psi$ is the reranking model, and $\eta_x$ is optional evidence such as metadata, retrieved context, pose features, or cross-modal features.

## Common Patterns

- Text RAG: retrieve chunks, then rerank by cross-encoder relevance or evidence quality.
- Paper triage: retrieve candidate papers, then rerank by topic fit, artifact availability, and novelty.
- Virtual screening: retrieve or dock candidates, then rerank by pose plausibility, affinity proxy, novelty, or liability filters.
- Multimodal search: retrieve by one modality, then rerank using aligned text-image, text-structure, or text-table evidence.

## Objective

A pairwise reranking loss can be written as:

$$
\mathcal{L}
=
-\log \sigma(s_1(q,x^+) - s_1(q,x^-))
$$

where $x^+$ should rank above $x^-$ for query $q$.

Listwise reranking can normalize scores inside the candidate set:

$$
P(x_i \mid q, C_k)
=
\frac{\exp(s_1(q,x_i))}
{\sum_{x_j \in C_k} \exp(s_1(q,x_j))}
$$

## Checks

- Does the first stage have enough recall for the reranker to help?
- Is the reranker evaluated only on retrieved candidates or on the full corpus?
- Does the metric emphasize early precision, enrichment, calibrated score, or downstream answer quality?
- Are first-stage retrieval errors counted separately from reranker errors?
- Does reranking use evidence available at deployment time?
- Are duplicates, homologs, scaffolds, or near-identical chunks removed before evaluation?

## Related

- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/tasks/similarity-search|Similarity search]]
- [[concepts/machine-learning/ranking|Ranking]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/metric-selection|Metric selection]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
