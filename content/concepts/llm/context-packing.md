---
title: Context Packing
tags:
  - llm
  - context
  - retrieval
---

# Context Packing

Context packing is the process of selecting, ordering, and formatting information for a model call. It decides what evidence and state enter the context window.

Given candidate context items $c_i$ with utility $u_i$ and token cost $l_i$, context packing resembles a constrained selection problem:

$$
\max_{S}
\sum_{i\in S} u_i
\quad
\text{subject to}
\quad
\sum_{i\in S} l_i \le B
$$

where $B$ is the available token budget after reserving instructions and output.

## Context Item Types

- User request and current goal.
- System or repository rules.
- Current file excerpts.
- Retrieved documents.
- Tool results.
- Prior decisions and summaries.
- Output schema or examples.

## Ordering Principles

- Put durable rules before task data.
- Put fresh task evidence near the action it supports.
- Separate untrusted content from instructions.
- Label sources and timestamps when relevance depends on freshness.
- Prefer concise excerpts over full dumps.

## Checks

- Does each context item support the next decision?
- Are source boundaries clear?
- Is any private or irrelevant information included?
- Is the model being asked to infer from missing evidence?
- Would a smaller context produce a more reliable answer?

## Related

- [[concepts/llm/token-budget|Token budget]]
- [[concepts/llm/context-window|Context window]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/memory-boundary|Memory boundary]]
