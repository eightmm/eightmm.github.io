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

## Checks

- What must be in context for the next decision?
- What can be retrieved later instead of loaded now?
- Are stale summaries overridden by fresh evidence?
- Are untrusted documents separated from instructions?
- Is important evidence being truncated?

## Related

- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/agent-memory|Agent memory]]
- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[concepts/architectures/attention|Attention]]
