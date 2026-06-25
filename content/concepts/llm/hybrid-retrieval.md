---
title: Hybrid Retrieval
tags:
  - llm
  - retrieval
  - rag
---

# Hybrid Retrieval

Hybrid retrieval combines lexical retrieval and embedding retrieval. It is useful when exact terms, symbols, acronyms, paper titles, identifiers, or code names matter, but semantic paraphrase also matters.

The combined score can be written as:

$$
s(q,d)
=
\lambda s_{\mathrm{lex}}(q,d)
+
(1-\lambda)s_{\mathrm{emb}}(q,d)
$$

where $s_{\mathrm{lex}}$ is a lexical score, $s_{\mathrm{emb}}$ is an embedding similarity score, and $\lambda\in[0,1]$ controls the mixture.

## Why It Matters

- Lexical retrieval catches exact names, formulas, filenames, and rare terms.
- Embedding retrieval catches paraphrases and conceptual similarity.
- Reranking can combine both candidate sets with a more expensive model.
- Metadata filters can enforce scope before ranking.

## Pipeline

$$
Q
\rightarrow
\{\text{lexical candidates}, \text{embedding candidates}\}
\rightarrow
\text{merge}
\rightarrow
\text{rerank}
\rightarrow
\text{context packing}
$$

## Checks

- Is the query term-sensitive, semantic, or both?
- Are lexical and embedding scores normalized before interpolation?
- Does the merge preserve source diversity rather than many near-duplicates?
- Is [[concepts/tasks/reranking|reranking]] evaluated separately from initial recall?
- Are metadata filters applied before retrieval when scope is known?
- Does the final context cite the exact retrieved evidence?

## Related

- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/tasks/reranking|Reranking]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/llm/context-packing|Context packing]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
