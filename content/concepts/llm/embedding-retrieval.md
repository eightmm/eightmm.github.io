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

Chunking changes the retrieval task. The retrieval unit can be:

$$
u_d \in \{\text{page}, \text{section}, \text{paragraph}, \text{table}, \text{code block}, \text{metadata}\}
$$

The unit should match the evidence granularity needed by downstream generation.

## Checks

- What is embedded: page, paragraph, chunk, title, code block, or metadata?
- Are chunks small enough to retrieve precisely but large enough to preserve context?
- Is retrieval evaluated with recall, precision, or downstream answer quality?
- Are stale or private documents excluded from public retrieval?
- Are embeddings versioned when the corpus or model changes?
- Does the retrieved unit contain enough context to support a claim?
- Are metadata filters used before vector similarity when scope matters?

## Related

- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
- [[concepts/machine-learning/ranking|Ranking]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/llm/evidence-grounded-generation|Evidence-grounded generation]]
- [[agents/core/agent-memory|Agent memory]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
