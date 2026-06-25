---
title: Chunking
tags:
  - llm
  - retrieval
  - rag
---

# Chunking

Chunking splits a document collection into retrieval units. In [[concepts/llm/retrieval-augmented-generation|RAG]], the chunk is often the item that is embedded, indexed, retrieved, and packed into context.

Given a document $d$, chunking produces:

$$
C(d)
=
\{c_1,\ldots,c_n\}
$$

where each chunk $c_i$ has text, metadata, source location, and token length $\ell_i$.

## Trade-off

Small chunks improve precision but can lose context. Large chunks preserve context but can retrieve irrelevant material and waste the [[concepts/llm/token-budget|token budget]].

The practical objective is:

$$
\max_C
\operatorname{support}(C, q)
\quad
\text{s.t.}
\quad
\sum_{c_i\in C_q}\ell_i \le B_{\mathrm{retrieval}}
$$

where $C_q$ is the retrieved chunk set for query $q$.

## Chunk Types

- Fixed-size token chunks: simple, but may cut sections or formulas.
- Paragraph chunks: good for prose, weak for tables and code.
- Section chunks: preserve headings and intent.
- Semantic chunks: split by topic or discourse boundary.
- Structured chunks: tables, equations, code blocks, figures, metadata, or paper sections.

## Checks

- Is the chunk unit aligned with the evidence needed by the answer?
- Does the chunk include source title, section, and stable location metadata?
- Are tables, equations, code blocks, and captions preserved as coherent units?
- Is overlap used deliberately rather than as a default patch?
- Does retrieval evaluation measure chunk recall separately from answer quality?
- Are private or unpublished chunks excluded from public indexes?

## Related

- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[concepts/llm/context-packing|Context packing]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/llm/evidence-grounded-generation|Evidence-grounded generation]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
