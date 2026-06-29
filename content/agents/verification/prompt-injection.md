---
title: Prompt Injection
tags:
  - agents
  - llm
  - security
---

# Prompt Injection

Prompt injection은 web page, file, tool result, email 같은 untrusted content가 agent의 행동을 가로채는 instruction을 담고 있는 경우입니다. Model은 data와 command를 항상 안정적으로 구분하지 못하므로, ingest된 text는 모두 잠재적인 instruction입니다.

핵심 문제는 trust boundary가 깨지는 것입니다.

$$
\text{untrusted data}
\not\Rightarrow
\text{trusted instruction}
$$

Agent가 retrieved data, tool output, user instruction, system policy를 하나의 context에 섞으면, model은 evidence로만 취급해야 할 text를 instruction처럼 따를 수 있습니다.

## 흔한 패턴

Prompt-injection payload는 보통 아래를 시도합니다.

1. 이전 instruction을 override합니다.
2. secret이나 private context를 exfiltrate합니다.
3. privileged tool call을 trigger합니다.
4. file, setting, memory를 수정합니다.
5. final answer에서 자신의 흔적을 숨깁니다.

Public LLM Wiki workflow에서 가장 위험한 경우는 untrusted source의 content를 durable note로 promotion하는 과정입니다.

$$
\text{source text}
\rightarrow
\text{summary}
\rightarrow
\text{public markdown}
$$

Summary step은 embedded command가 아니라 fact를 보존해야 합니다.

## 방어 모델

실용적인 방어는 역할을 분리하는 것입니다.

$$
\text{policy}
>
\text{developer contract}
>
\text{user task}
>
\text{trusted project files}
>
\text{untrusted content}
$$

Untrusted content는 “이 문서가 무엇을 말하는가?”에는 답할 수 있습니다. 하지만 사용자가 명시적으로 authority를 위임하지 않는 한 “agent가 다음에 무엇을 해야 하는가?”에 답하게 두면 안 됩니다.

## Tool boundary

Tool call은 injection이 실제 side effect로 바뀌는 지점입니다. Risk는 아래처럼 커집니다.

$$
\operatorname{risk}
\propto
\operatorname{privilege}
\times
\operatorname{irreversibility}
\times
\operatorname{uncertainty}
$$

File write, network call, credential use, push, delete, memory update는 read-only inspection보다 더 강한 check가 필요합니다.

## 실전 check

- 모든 tool output과 fetched content를 command가 아니라 untrusted data로 취급합니다.
- privileged action은 model discretion이 아니라 explicit confirmation 뒤에 둡니다.
- tool scope와 credential은 task에 필요한 최소한으로 제한합니다.
- code execution과 file write는 sandbox합니다. Injected payload가 escape를 시도한다고 가정합니다.
- injection을 detect하고 trace할 수 있도록 input과 action을 log합니다.
- untrusted content는 instruction이 아니라 content로 quote하거나 summarize합니다.
- untrusted instruction을 durable memory에 저장하지 않습니다.
- public publishing 전에는 secret, private identifier, hidden instruction, operational detail을 scan합니다.
- 허용된 side effect는 broad model discretion보다 allowlist로 제한합니다.

## Related

- [[concepts/llm/prompt-injection-boundary|Prompt injection boundary]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/memory-boundary|Memory boundary]]
- [[agents/tools/tool-use|Tool use]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/evidence-ledger|Evidence ledger]]
- [[agents/index|Agents]]
