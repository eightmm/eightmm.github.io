---
title: Embedding Retrieval
tags:
  - llm
  - retrieval
  - representation-learning
---

# Embedding Retrieval

Embedding retrieval maps queries and documents into vectors, then retrieves nearby documents according to a similarity score.

For query embedding $z_q$ and document embedding $z_d$:

$$
s(q,d)
=
\frac{z_q^\top z_d}
{\lVert z_q\rVert_2\lVert z_d\rVert_2}
$$

Top-k retrieval returns:

$$
D_q
=
\operatorname{TopK}_{d\in\mathcal{D}}
s(q,d)
$$

## Checks

- What is embedded: page, paragraph, chunk, title, code block, or metadata?
- Are chunks small enough to retrieve precisely but large enough to preserve context?
- Is retrieval evaluated with recall, precision, or downstream answer quality?
- Are stale or private documents excluded from public retrieval?
- Are embeddings versioned when the corpus or model changes?

## Related

- [[concepts/machine-learning/representation-learning|Representation learning]]
- [[concepts/machine-learning/ranking|Ranking]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[agents/agent-memory|Agent memory]]
- [[agents/llm-wiki|LLM Wiki]]
