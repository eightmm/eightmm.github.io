---
title: Retrieval-Augmented Generation
tags:
  - llm
  - retrieval
  - rag
---

# Retrieval-Augmented Generation

Retrieval-augmented generation (RAG) combines a language model with external documents retrieved at query time. The model generates from both the query and retrieved evidence.

A simple RAG pipeline is:

$$
D_q = \operatorname{Retrieve}(q, \mathcal{D})
$$

$$
y \sim p_\theta(y \mid q, D_q)
$$

where $q$ is a query, $\mathcal{D}$ is the document collection, and $D_q$ is the retrieved subset.

RAG has two separable quality gates:

$$
\text{retrieval quality}
\ne
\text{answer faithfulness}
$$

Good retrieval can still produce an unsupported answer if generation ignores or distorts evidence.

In practical systems, retrieval often includes [[concepts/llm/query-rewriting|query rewriting]], [[concepts/llm/chunking|chunking]], [[concepts/llm/hybrid-retrieval|hybrid retrieval]], [[concepts/tasks/reranking|reranking]], and [[concepts/llm/context-packing|context packing]].

## Checks

- Are retrieved documents relevant, current, and source-attributed?
- What is the retrieval unit: document, section, paragraph, chunk, table, or code block?
- Is query rewriting improving recall without changing intent?
- Is retrieval evaluated separately from generation?
- Are untrusted retrieved documents protected against prompt injection?
- Does the answer cite or link the evidence it used?
- Is the corpus public, private, or mixed?
- Are unsupported generated claims marked or removed?

## Related

- [[concepts/llm/evidence-grounded-generation|Evidence-grounded generation]]
- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[concepts/llm/chunking|Chunking]]
- [[concepts/llm/hybrid-retrieval|Hybrid retrieval]]
- [[concepts/llm/query-rewriting|Query rewriting]]
- [[concepts/llm/context-packing|Context packing]]
- [[concepts/llm/prompt-injection-boundary|Prompt injection boundary]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[agents/verification/prompt-injection|Prompt injection]]
- [[agents/verification/verification-loop|Verification loop]]
- [[papers/systems/index|Systems papers]]
