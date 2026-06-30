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

Because the two scores often have different scales, practical hybrid retrieval usually normalizes or ranks before combining:

$$
s'(q,d)
=
\lambda\, \operatorname{norm}(s_{\mathrm{lex}}(q,d))
+
(1-\lambda)\, \operatorname{norm}(s_{\mathrm{emb}}(q,d))
$$

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

## Candidate Merge

Hybrid retrieval is often more robust when lexical and dense retrieval produce separate candidate sets first.

$$
C
=
\operatorname{TopK}_{\mathrm{lex}}(q)
\cup
\operatorname{TopK}_{\mathrm{emb}}(q)
$$

Then a merge or reranker orders $C$.

| Merge method | Use when | Risk |
| --- | --- | --- |
| score interpolation | scores are calibrated or normalized | one score family dominates |
| rank fusion | scales are not comparable | loses magnitude information |
| reciprocal rank fusion | robust simple baseline | can overvalue shallow matches |
| learned reranker | enough labels or strong cross-encoder exists | slower and harder to debug |

Reciprocal rank fusion is a common simple form:

$$
s_{\mathrm{RRF}}(d)
=
\sum_{r\in\mathcal{R}}
\frac{1}{k_0 + \operatorname{rank}_r(d)}
$$

where $\mathcal{R}$ is the set of retrieval runs.

## When Hybrid Helps

| Query type | Lexical signal | Embedding signal |
| --- | --- | --- |
| paper title or method name | exact phrase | related discussion |
| code symbol or file path | exact token | surrounding explanation |
| formula or acronym | literal match | expanded meaning |
| broad concept question | weak | semantic neighborhoods |
| public wiki navigation | page title and aliases | conceptual adjacency |

Hybrid retrieval is useful for LLM Wiki because a user may search for `Xid`, `RMSD`, `JEPA`, or `Slurm QOS`, where exact terms and semantic explanation both matter.

## Debugging Slices

| Failure | Check |
| --- | --- |
| exact document missing | lexical candidate set too small or tokenization issue |
| semantically right but wrong version | metadata/date filter missing |
| many duplicates | chunking too fine or merge lacks diversity |
| top result is related but not evidential | reranker objective too broad |
| rare term ignored | embedding model did not preserve identifier |

## Checks

- Is the query term-sensitive, semantic, or both?
- Are lexical and embedding scores normalized before interpolation?
- Does the merge preserve source diversity rather than many near-duplicates?
- Is [[concepts/tasks/reranking|reranking]] evaluated separately from initial recall?
- Are metadata filters applied before retrieval when scope is known?
- Does the final context cite the exact retrieved evidence?
- Are lexical recall, embedding recall, and reranker quality evaluated separately?
- Does merge logic preserve both exact-match and semantic candidates?

## Related

- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/tasks/reranking|Reranking]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/llm/context-packing|Context packing]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
