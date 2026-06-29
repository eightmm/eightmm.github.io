---
title: Memory Boundary
tags:
  - agents
  - memory
  - privacy
---

# Memory Boundary

Memory boundary는 agent가 task를 넘어서 무엇을 저장, recall, reuse할 수 있는지 정의합니다. 유용한 long-term context가 stale assumption, privacy leak, task drift의 원인이 되지 않도록 막는 역할입니다.

Memory는 아래처럼 나눌 수 있습니다.

$$
M
=
M_{\mathrm{working}}
\cup
M_{\mathrm{durable}}
\cup
M_{\mathrm{external}}
$$

여기서 working memory는 task-local이고, durable memory는 session을 넘어 persist되며, external memory는 file, doc, database, search에서 retrieve되는 정보입니다.

## Boundary Questions

- 무엇을 persist해도 안전한가?
- 무엇은 task-local로 남아야 하는가?
- 무엇은 절대 저장하면 안 되는가?
- 어떤 fact는 reuse 전에 revalidation이 필요한가?
- durable memory를 update할 권한은 누구에게 있는가?

## Public Wiki Rule

이 블로그의 public durable note에는 general concept, public workflow, sanitized guidance만 저장합니다. Private infrastructure detail, credential, internal task name, unpublished result, collaborator-specific information은 저장하지 않습니다.

## Stale Memory Failure

Agent는 old context를 current fact처럼 취급해서 실패할 수 있습니다.

$$
\operatorname{risk}
\propto
\operatorname{age}(m)
\times
\operatorname{impact}(m)
\times
(1-\operatorname{verification}(m))
$$

이 식은 실제 calibrated metric이 아니라 운영 습관을 표현한 것입니다. 오래됐고 impact가 큰 memory일수록 더 강한 verification이 필요합니다.

## Checks

- remembered fact가 public, current, relevant한가?
- source가 action에 충분히 authoritative한가?
- 이 fact를 저장하면 private information이 leak될 수 있는가?
- task에 durable memory가 필요한가, local note만 필요한가?
- wrong memory에 대한 deletion 또는 correction path가 있는가?

## Related

- [[agents/core/agent-memory|Agent memory]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[logs/sanitization-checklist|Sanitization checklist]]
- [[concepts/llm/context-window|Context window]]
