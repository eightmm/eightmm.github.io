---
title: Query Rewriting
tags:
  - llm
  - retrieval
  - rag
---

# Query Rewriting

Query rewriting transforms a user request into one or more retrieval queries. It is common in RAG systems because the user's natural-language question is not always the best search query.

A rewriting function maps:

$$
R(q, h, s)
\rightarrow
\{q'_1,\ldots,q'_k\}
$$

where $q$ is the current query, $h$ is conversation history, and $s$ is task state.

## Rewrite Types

- Clarifying rewrite: make implicit references explicit.
- Decomposition: split a complex question into subqueries.
- Expansion: add synonyms, acronyms, or related concepts.
- Constraint extraction: turn a request into filters such as date, source, domain, modality, or entity.
- HyDE-style rewrite: generate a hypothetical answer or passage, then retrieve against it.

## Risks

Query rewriting can improve recall but can also drift from the user's intent:

$$
\operatorname{drift}(q,q')
=
1 - \operatorname{sim}(\operatorname{intent}(q), \operatorname{intent}(q'))
$$

The rewrite should be treated as a retrieval aid, not as a new task definition.

## Checks

- Does the rewrite preserve the original task and constraints?
- Are filters extracted from evidence or invented?
- Is the rewritten query visible in logs or artifacts for debugging?
- Are multiple rewrites merged without flooding the context window?
- Is retrieval measured before and after rewriting?
- Can the system fall back to the original query?

## Related

- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/llm/hybrid-retrieval|Hybrid retrieval]]
- [[concepts/llm/context-packing|Context packing]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/tasks/question-answering|Question answering]]
- [[agents/core/context-engineering|Context engineering]]
