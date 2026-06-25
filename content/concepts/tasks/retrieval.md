---
title: Retrieval
tags:
  - tasks
  - retrieval
  - representation-learning
---

# Retrieval

Retrieval returns relevant items from a collection instead of predicting a fixed label. It appears in search engines, embedding databases, [[concepts/llm/retrieval-augmented-generation|RAG]], molecule similarity search, paper triage, and nearest-neighbor recommendation.

Given a query $q$ and candidate set $\mathcal{D}$, retrieval ranks candidates by a scoring function:

$$
s(q, d) = \operatorname{sim}(f_\theta(q), g_\phi(d))
$$

The top-$k$ result set is:

$$
\operatorname{TopK}(q) = \operatorname{arg\,topk}_{d \in \mathcal{D}} s(q,d)
$$

The task specification should state the corpus, relevance definition, and whether retrieval is final output or only a recall stage for a reranker or generator.

## Common Designs

- Sparse lexical retrieval with term matching.
- Dense retrieval using learned embeddings.
- Hybrid retrieval combining sparse and dense scores.
- Two-stage systems: retrieval first, reranking second.

## Metrics

- Recall@$k$: whether relevant items appear in the retrieved set.
- Precision@$k$: how many retrieved items are relevant.
- MRR: reciprocal rank of the first relevant item.
- nDCG: ranking quality with graded relevance.

## Checks

- What counts as relevant?
- Is the candidate corpus fixed or changing?
- Are near-duplicates split correctly?
- Does retrieval optimize recall for downstream reranking, or precision for direct use?
- Are embedding updates synchronized with the index?
- Is freshness, citation support, or provenance required?

## Related

- [[concepts/tasks/task-specification|Task specification]]
- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/machine-learning/ranking|Ranking]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[concepts/evaluation/metric|Metric]]
