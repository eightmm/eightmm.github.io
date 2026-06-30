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

In practice, retrieval is a two-part contract:

$$
q
\rightarrow
\{u_i\}_{i=1}^{k}
\rightarrow
\text{answer evidence}
$$

The system must retrieve units that are not only similar to the query, but sufficient to support the downstream claim.

[[concepts/llm/chunking|Chunking]] changes the retrieval task. The retrieval unit can be:

$$
u_d \in \{\text{page}, \text{section}, \text{paragraph}, \text{table}, \text{code block}, \text{metadata}\}
$$

The unit should match the evidence granularity needed by downstream generation.

## Embedding Space

Embedding retrieval depends on what the embedding model was trained to consider similar.

| Similarity type | Good for | Failure mode |
| --- | --- | --- |
| Semantic paraphrase | natural language questions and explanations | misses exact identifiers |
| Domain concept | related scientific or technical concepts | retrieves broad background instead of evidence |
| Instruction-query alignment | question-answer retrieval | weak for raw tables, code, formulas |
| Code/text mixed space | code search and API examples | can over-rank names without semantic support |

The embedding function should match the corpus and retrieval goal:

$$
z_q = f_\phi(q), \quad z_d=f_\phi(d)
$$

Changing $f_\phi$ changes the meaning of nearest neighbors, so embedding model, corpus version, and chunking policy should be versioned together.

## Retrieval Evaluation

Retrieval quality is not the same as answer quality, but bad retrieval usually bounds answer quality.

For a query $q$ with relevant set $R_q$:

$$
\operatorname{Recall@k}(q)
=
\frac{|D_q^{(k)} \cap R_q|}{|R_q|}
$$

$$
\operatorname{Precision@k}(q)
=
\frac{|D_q^{(k)} \cap R_q|}{k}
$$

| Metric | Use when | Trap |
| --- | --- | --- |
| Recall@k | missing evidence is costly | high recall can still include noisy context |
| Precision@k | context window is tight | may punish useful supporting background |
| MRR | one best answer document matters | weak when evidence is distributed |
| nDCG | graded relevance exists | relevance labels can be subjective |
| answer grounding | RAG output is evaluated end-to-end | may hide retrieval failures behind generation |

## Filtering and Metadata

Vector similarity should not replace scope filters. If the task has known constraints, filter before ranking.

| Filter | Why |
| --- | --- |
| source type | paper, code, docs, private note, public note have different trust |
| date or version | stale docs can be semantically close but wrong |
| section or field | title, abstract, method, table, code block serve different claims |
| privacy boundary | private or internal documents should not enter public output |
| domain | AI, infra, computational biology, agents may share terms with different meanings |

The useful pattern is:

$$
\mathcal{D}_{\mathrm{scope}}
=
\{d\in\mathcal{D}: \operatorname{filter}(d,q)=1\}
$$

then rank only inside $\mathcal{D}_{\mathrm{scope}}$.

## Checks

- What is embedded: page, paragraph, chunk, title, code block, or metadata?
- Are chunks small enough to retrieve precisely but large enough to preserve context?
- Is retrieval evaluated with recall, precision, or downstream answer quality?
- Are stale or private documents excluded from public retrieval?
- Are embeddings versioned when the corpus or model changes?
- Does the retrieved unit contain enough context to support a claim?
- Are metadata filters used before vector similarity when scope matters?
- Is retrieval being evaluated separately from generation?
- Does the top result support the exact claim, or only the general topic?

## Related

- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/llm/chunking|Chunking]]
- [[concepts/llm/hybrid-retrieval|Hybrid retrieval]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
- [[concepts/machine-learning/ranking|Ranking]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/llm/evidence-grounded-generation|Evidence-grounded generation]]
- [[agents/core/agent-memory|Agent memory]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
