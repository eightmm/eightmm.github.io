---
title: Similarity Search
tags:
  - tasks
  - retrieval
  - representation-learning
---

# Similarity Search

Similarity search retrieves items that are close to a query under a chosen representation and similarity function. It is a task pattern shared by text retrieval, image search, molecule search, protein sequence search, nearest-neighbor baselines, and memory lookup.

The basic form is:

$$
z_q = f_\theta(q), \qquad z_i = g_\phi(x_i)
$$

$$
s(q, x_i) = \operatorname{sim}(z_q, z_i)
$$

$$
\operatorname{NN}_k(q)
=
\operatorname{arg\,topk}_{x_i \in \mathcal{D}}
s(q, x_i)
$$

where $q$ is a query, $x_i$ is a candidate item, $z_q$ and $z_i$ are representations, $\mathcal{D}$ is the corpus, and $s$ is a similarity score.

## Representation Choices

| Domain | Representation | Common similarity |
| --- | --- | --- |
| Text | dense embedding, sparse vector | cosine, dot product, BM25 score |
| Image | patch/global embedding | cosine, Euclidean distance |
| Molecule | fingerprint, graph embedding, shape feature | Tanimoto, cosine, RMSD-like distance |
| Protein sequence | residue embedding, k-mer, alignment feature | cosine, identity, alignment score |
| 3D structure | coordinates, distances, geometric graph | RMSD, distance-map score, learned score |

Changing the representation changes the task. A near neighbor under text embeddings is not necessarily a near neighbor under citation graph, timestamp, molecular scaffold, or 3D shape.

## Metrics

- Recall@$k$: whether relevant items appear in the retrieved neighborhood.
- Precision@$k$: how many returned neighbors are relevant.
- MRR: rank of the first relevant item.
- nDCG: graded relevance under a ranked list.
- Coverage: fraction of queries for which the index can return usable candidates.

## Checks

- What representation defines closeness?
- Is the similarity score symmetric or query-conditioned?
- Is the index exact, approximate, filtered, or hybrid?
- Are near-duplicates and homologs handled by the split?
- Is search used as a final task or as a recall stage before [[concepts/tasks/reranking|Reranking]]?
- Is a simple nearest-neighbor baseline strong enough to explain the result?

## Related

- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/tasks/reranking|Reranking]]
- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
- [[concepts/molecular-modeling/molecular-similarity|Molecular similarity]]
- [[concepts/protein-modeling/sequence-identity-clustering|Sequence identity clustering]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/leakage|Leakage]]
