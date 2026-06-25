---
title: Prompt Injection Boundary
tags:
  - llm
  - security
  - agents
---

# Prompt Injection Boundary

A prompt injection boundary separates trusted instructions from untrusted content. It is a workflow rule, not a model capability: the model may read untrusted text, but it should not treat it as authority.

Context can be divided into:

$$
C
=
C_{\mathrm{trusted}}
\cup
C_{\mathrm{untrusted}}
\cup
C_{\mathrm{generated}}
$$

Only $C_{\mathrm{trusted}}$ should define goals, permissions, and rules. $C_{\mathrm{untrusted}}$ should be treated as data.

## Untrusted Sources

- Web pages.
- Retrieved documents.
- Uploaded files.
- Tool output.
- Emails, issues, comments, or chat transcripts.
- Model-generated intermediate drafts.

## Boundary Practices

- Label untrusted content explicitly.
- Never let retrieved text override system, repository, or user instructions.
- Restrict side-effecting tools.
- Validate tool arguments and generated code before execution.
- Keep secrets and credentials out of model-visible context.
- Prefer source quotes or links for evidence, but do not copy private material.

## Checks

- What content in context is untrusted?
- Could this text contain instructions to the model?
- Can the model trigger privileged tools from this content?
- Are sensitive values exposed to the model or logs?
- Is the final answer grounded in verified evidence rather than injected claims?

## Related

- [[agents/verification/prompt-injection|Prompt injection]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[concepts/llm/tool-calling|Tool calling]]
- [[concepts/llm/context-packing|Context packing]]
- [[agents/tools/tool-contract|Tool contract]]
