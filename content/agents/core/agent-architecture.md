---
title: Agent Architecture
tags:
  - agents
  - architecture
  - workflows
---

# Agent Architecture

Agent architecture는 model call을 workflow로 바꾸는 component를 설명합니다. 유용한 agent는 LLM 하나가 아니라 state, tool, memory, policy, verification을 가진 loop입니다.

간단한 agent는 아래처럼 쓸 수 있습니다.

$$
a_t = \pi_\theta(s_t, c_t, m_t)
$$

여기서 $s_t$는 task state, $c_t$는 current context, $m_t$는 retrieved memory, $a_t$는 next action입니다.

## Component

- Model: plan, tool call, edit, answer를 생성합니다.
- State: goal, constraint, evidence, progress를 추적합니다.
- Context: current call에서 보이는 selected information입니다.
- Tools: external state를 inspect하거나 change하는 action입니다.
- Memory: context window 밖의 durable 또는 retrieved information입니다.
- Verifier: artifact가 correct한지 확인합니다.
- Human boundary: approval 또는 judgment가 필요한 위치를 정의합니다.

## 확인할 것

- 어떤 state가 conversation history에 숨어 있지 않고 explicit한가?
- 어떤 tool이 side effect를 만들 수 있는가?
- 각 side effect를 무엇으로 verify하는가?
- 무엇이 loop를 멈추게 하는가?
- 어떤 decision에 human review가 필요한가?

## Related

- [[agents/core/agent-loop|Agent loop]]
- [[agents/core/agent-state|Agent state]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/verification/verification-loop|Verification loop]]
