---
title: Chat and Prompting
tags:
  - agents
  - prompting
---

# Chat and Prompting

Chat은 가장 기본적인 LLM 인터페이스입니다. agent 관점에서는 사용자의 요청, 시스템 규칙, 과거 대화, 첨부 자료가 하나의 context로 합쳐지고, 모델은 다음 응답 또는 다음 행동을 선택합니다.

$$
y_t \sim p_\theta(y \mid I, C_t, x_t)
$$

여기서 $I$는 instruction, $C_t$는 current context, $x_t$는 user request입니다.

Chat은 가장 가벼운 interface지만, 모든 agent workflow의 입구이기도 합니다. 따라서 좋은 chat 사용은 “한 번에 맞는 답을 받기”보다 task state를 명확하게 만들고 다음 action을 좁히는 일에 가깝습니다.

## 좋은 프롬프트의 형태

| 구성 요소 | 역할 |
| --- | --- |
| Goal | 무엇을 끝내야 하는지 |
| Context | 판단에 필요한 배경 |
| Constraints | 하지 말아야 할 것, 형식, 공개 범위 |
| Output | 원하는 산출물의 형태 |
| Verification | 답을 어떻게 확인할지 |

프롬프트는 길수록 좋은 것이 아니라, 다음 결정을 내리기에 충분해야 합니다. 같은 요청이라도 연구 요약, 코드 수정, 논문 비교, 공개 블로그 작성은 필요한 context가 다릅니다.

## Prompt Modes

| Mode | 목적 | 좋은 출력 |
| --- | --- | --- |
| Explain | 개념, 수식, code path 이해 | definition, example, limitation |
| Decide | 선택지 비교 | criteria, tradeoff, recommendation |
| Transform | 글, 표, 코드, note 변환 | diff-friendly rewrite |
| Plan | 큰 작업을 작은 step으로 나눔 | step, evidence, risk |
| Review | 결과물의 결함 찾기 | severity, location, fix direction |

## Agent와의 차이

단순 chat은 응답을 생성하고 끝납니다. Agent workflow는 응답 이후에도 도구를 호출하거나, 파일을 수정하거나, 결과를 검증할 수 있습니다.

$$
\text{chat}:\ x \rightarrow y
$$

$$
\text{agent}:\ x \rightarrow a_1 \rightarrow o_1 \rightarrow a_2 \rightarrow o_2 \rightarrow y
$$

## 확인할 것

- 대화의 목적이 answer, decision, artifact, action 중 무엇인가?
- model에게 필요한 source가 실제로 제공됐는가?
- 답이 검증 가능한 claim과 추론으로 나뉘는가?
- 다음 agent action으로 넘길 수 있을 만큼 constraints가 명확한가?

## Related

- [[concepts/llm/prompting|Prompting]]
- [[agents/core/prompt-engineering|Prompt engineering]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/agent-loop|Agent loop]]
