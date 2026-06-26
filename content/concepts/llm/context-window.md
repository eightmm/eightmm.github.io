---
title: Context Window
tags:
  - llm
  - context
---

# Context Window

The context window is the token budget a model can attend to during one call. It contains the prompt, instructions, retrieved evidence, conversation state, tool results, and requested output.

If $C$ is the maximum context length:

$$
|x_{\mathrm{input}}| + |x_{\mathrm{output}}| \le C
$$

Context is not memory by itself. It is the temporary working set for one model invocation.

The context should be allocated by role:

$$
C =
C_{\mathrm{instructions}}
+ C_{\mathrm{state}}
+ C_{\mathrm{evidence}}
+ C_{\mathrm{tools}}
+ C_{\mathrm{output}}
$$

When context is tight, prioritize current evidence over stale summaries.

## Attention Cost

For standard full attention, compute and memory scale roughly with context length:

$$
\operatorname{cost}
\propto
O(C^2)
$$

This is why long context is not the same as free memory. Even if a model can accept a long prompt, irrelevant context can increase latency, cost, and distraction.

## Context Allocation

A practical allocation can be treated as an optimization problem:

$$
\max_{S\subseteq E}
\operatorname{utility}(S)
\quad
\text{s.t.}
\quad
\operatorname{tokens}(S)
\le
C_{\mathrm{evidence}}
$$

where $E$ is candidate evidence and $S$ is the selected subset. Good context packing selects evidence that is current, authoritative, relevant, and non-duplicative.

## Recency and Position

LLMs can be sensitive to where information appears. Important constraints should be placed in stable instruction or task sections, while long evidence blocks should be structured with clear labels:

$$
\text{instruction}
\ne
\text{untrusted evidence}
\ne
\text{tool output}
$$

This separation matters for [[concepts/llm/prompt-injection-boundary|prompt injection boundaries]].

## Memory Boundary

Context is temporary. Memory is persisted or retrieved across calls:

$$
\text{context}
\subset
\text{current invocation},
\qquad
\text{memory}
\subset
\text{cross-invocation state}
$$

Do not treat a context summary as authoritative when current source files, tool output, or user instructions contradict it.

## Checks

- What must be in context for the next decision?
- What can be retrieved later instead of loaded now?
- Are stale summaries overridden by fresh evidence?
- Are untrusted documents separated from instructions?
- Is important evidence being truncated?
- Is output budget reserved before loading more evidence?
- Are repeated summaries crowding out direct evidence?
- Is the context ordered so that high-priority instructions are easy to distinguish from data?
- Is the task using context as a substitute for durable memory when retrieval would be better?

## Related

- [[concepts/llm/token-budget|Token budget]]
- [[concepts/llm/context-packing|Context packing]]
- [[concepts/llm/evidence-grounded-generation|Evidence-grounded generation]]
- [[concepts/llm/prompt-injection-boundary|Prompt injection boundary]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/agent-memory|Agent memory]]
- [[agents/core/memory-boundary|Memory boundary]]
- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[concepts/architectures/attention|Attention]]
