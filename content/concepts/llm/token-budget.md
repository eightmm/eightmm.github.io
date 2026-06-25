---
title: Token Budget
tags:
  - llm
  - context
  - systems
---

# Token Budget

Token budget is the allocation of context length across instructions, task state, retrieved evidence, tool results, examples, and expected output. It is the practical version of the context window constraint.

If $C$ is the model context limit:

$$
B_{\mathrm{system}}
+ B_{\mathrm{task}}
+ B_{\mathrm{memory}}
+ B_{\mathrm{retrieval}}
+ B_{\mathrm{tools}}
+ B_{\mathrm{output}}
\le
C
$$

where each $B$ is a token allocation.

## Why It Matters

- Too much retrieved text can push out the actual instruction.
- Too many examples can remove fresh evidence.
- Large tool outputs can hide the relevant signal.
- Under-budgeted output space can truncate reasoning, code, or citations.
- Long context increases cost and can make attention to key evidence less reliable.

## Allocation Strategy

- Reserve output budget before filling input context.
- Keep durable rules short and link to source documents.
- Retrieve only the evidence needed for the next step.
- Summarize tool output into structured facts when possible.
- Prefer iterative retrieval over loading everything.

## Checks

- What information must be present for this call?
- What can be fetched later?
- Is output budget explicitly reserved?
- Are stale summaries competing with fresh evidence?
- Are untrusted documents separated from instructions?

## Related

- [[concepts/llm/context-window|Context window]]
- [[concepts/llm/context-packing|Context packing]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/agent-memory|Agent memory]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
