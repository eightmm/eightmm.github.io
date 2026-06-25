---
title: Context Engineering
tags:
  - agents
  - llm
  - context
---

# Context Engineering

Context engineering is the practice of giving an agent the right information at the right time. Too little context causes blind guesses; too much context causes distraction, stale assumptions, and missed constraints.

The working context can be viewed as:

$$
C_t
=
G
\cup S_t
\cup E_t
\cup R_t
$$

where $G$ is the goal, $S_t$ is current working state, $E_t$ is evidence from tools or files, and $R_t$ is retrieved memory or references.

## Principles

- Put durable rules in stable documents, not repeated prompts.
- Retrieve evidence on demand instead of loading everything.
- Prefer primary files, command output, and rendered artifacts over memory.
- Keep summaries explicit about uncertainty and missing evidence.
- Remove secrets and private operational details from shared context.

## Checks

- Does the context include the actual current state?
- Are stale summaries overridden by fresh evidence?
- Are constraints and success criteria visible before action?
- Are untrusted documents separated from instructions?

## Related

- [[concepts/llm/context-window|Context window]]
- [[concepts/llm/in-context-learning|In-context learning]]
- [[agents/core/agent-memory|Agent memory]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/verification/prompt-injection|Prompt injection]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
