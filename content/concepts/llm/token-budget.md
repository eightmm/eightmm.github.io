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

For an LLM Wiki workflow, the useful budget is not only total length but the ratio between evidence and overhead:

$$
B_{\mathrm{evidence}}
\le
C - (B_{\mathrm{instructions}} + B_{\mathrm{state}} + B_{\mathrm{output}})
$$

If $B_{\mathrm{evidence}}$ is too small, the model has no grounding. If overhead is too small, the model may ignore constraints, task state, or output requirements.

## Why It Matters

- Too much retrieved text can push out the actual instruction.
- Too many examples can remove fresh evidence.
- Large tool outputs can hide the relevant signal.
- Under-budgeted output space can truncate reasoning, code, or citations.
- Long context increases cost and can make attention to key evidence less reliable.

## Budget Map

| Budget Area | Keep | Trim First |
| --- | --- | --- |
| Instructions | durable rules, current objective, constraints | repeated policy text already linked elsewhere |
| Task state | current files, decisions, open questions | stale summaries that conflict with current source |
| Evidence | exact relevant snippets, citations, command outputs | broad dumps and duplicate sources |
| Tool results | structured facts and errors | full logs without relevance filtering |
| Examples | only examples that shape output format | old examples with different constraints |
| Output | enough room for final answer or patch reasoning | unnecessary verbosity |

## Allocation Strategy

- Reserve output budget before filling input context.
- Keep durable rules short and link to source documents.
- Retrieve only the evidence needed for the next step.
- Summarize tool output into structured facts when possible.
- Prefer iterative retrieval over loading everything.
- Drop duplicated evidence before dropping the current task constraints.

## Checks

- What information must be present for this call?
- What can be fetched later?
- Is output budget explicitly reserved?
- Are stale summaries competing with fresh evidence?
- Are untrusted documents separated from instructions?
- Is the retrieved context tied to the current question rather than the whole topic?
- Can a later verifier recover the cited evidence from stable paths or links?

## Related

- [[concepts/llm/context-window|Context window]]
- [[concepts/llm/context-packing|Context packing]]
- [[concepts/llm/chunking|Chunking]]
- [[concepts/llm/evidence-grounded-generation|Evidence-grounded generation]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/agent-memory|Agent memory]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
