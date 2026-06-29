---
title: Agent Core
tags:
  - agents
  - core
---

# Agent Core

Agent core는 LLM agent를 이루는 기본 부품을 정리합니다. 여기서는 제품명이 아니라 prompt, context, loop, state, memory, planning, harness처럼 대부분의 agent 시스템에 공통으로 들어가는 구조를 봅니다.

Agent는 제한된 action space 위에서 움직이는 stateful policy로 볼 수 있습니다.

$$
a_t = \pi_\theta(o_t, s_t, c_t, m_t)
$$

여기서 $o_t$는 현재 observation, $s_t$는 명시적 task state, $c_t$는 선택된 context, $m_t$는 검색된 memory, $a_t$는 답변, 계획, tool call, 파일 수정, human input 요청 같은 action입니다.

중요한 것은 무엇을 명시적으로 관리하느냐입니다.

$$
\text{agent}
=
(\text{model},\ \text{state},\ \text{context},\ \text{memory},\ \text{tools},\ \text{verifier})
$$

이 부품들이 암묵적으로 섞이면 agent는 동작하더라도 디버깅과 검증이 어려워집니다.

## 읽기 경로

1. [[agents/core/agent-architecture|Agent architecture]]에서 전체 부품을 봅니다.
2. [[agents/core/agent-loop|Agent loop]]에서 observe-plan-act-verify 반복을 봅니다.
3. [[agents/core/prompt-engineering|Prompt engineering]], [[agents/core/context-engineering|Context engineering]], [[agents/core/agent-state|Agent state]]로 현재 run에 필요한 정보를 정합니다.
4. [[agents/core/planning|Planning]]과 [[agents/core/task-decomposition|Task decomposition]]으로 큰 작업을 검증 가능한 단위로 나눕니다.
5. [[agents/core/agent-memory|Agent memory]]와 [[agents/core/memory-boundary|Memory boundary]]로 장기 기억과 현재 작업 맥락을 분리합니다.
6. [[agents/core/harness-engineering|Harness engineering]]으로 tool, permission, verifier를 감싸는 실행층을 봅니다.

## Notes

| 그룹 | 노트 |
| --- | --- |
| Architecture | [[agents/core/agent-architecture|Agent architecture]], [[agents/core/agent-operating-contract|Agent operating contract]] |
| Loop and state | [[agents/core/agent-loop|Agent loop]], [[agents/core/agent-environment|Agent environment]], [[agents/core/action-space|Action space]], [[agents/core/agent-state|Agent state]] |
| Prompt and context | [[agents/core/prompt-engineering|Prompt engineering]], [[agents/core/context-engineering|Context engineering]] |
| Planning | [[agents/core/planning|Planning]], [[agents/core/task-decomposition|Task decomposition]] |
| Memory | [[agents/core/agent-memory|Agent memory]], [[agents/core/memory-boundary|Memory boundary]] |
| Harness | [[agents/core/harness-engineering|Harness engineering]] |

## Checks

- 현재 goal은 무엇이고 어디에 저장되는가?
- 어떤 observation이 agent state를 갱신할 수 있는가?
- 어떤 action이 read-only, side-effecting, human-gated인가?
- 다음 결정에 필요한 context와 noise는 무엇인가?
- 어떤 memory가 durable하고, 어떤 것은 한 run 안에만 남아야 하는가?
- 어떤 evidence가 loop를 멈추게 하는가?

## Related

- [[agents/index|Agents]]
- [[agents/features/index|Agent features]]
- [[agents/tools/index|Agent tools]]
- [[agents/verification/index|Agent verification]]
- [[concepts/llm/index|LLM concepts]]
