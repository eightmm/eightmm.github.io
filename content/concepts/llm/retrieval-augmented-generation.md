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

## Checks

- Are retrieved documents relevant, current, and source-attributed?
- Is retrieval evaluated separately from generation?
- Are untrusted retrieved documents protected against prompt injection?
- Does the answer cite or link the evidence it used?
- Is the corpus public, private, or mixed?

## Related

- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[agents/verification/prompt-injection|Prompt injection]]
- [[agents/verification/verification-loop|Verification loop]]
- [[papers/systems/index|Systems papers]]
