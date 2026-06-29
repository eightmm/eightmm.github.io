---
title: Tool Result Handling
tags:
  - agents
  - tools
  - verification
---

# Tool Result Handling

Tool result handling은 tool output을 instruction이 아니라 evidence로 취급하는 실천입니다. Agent는 tool이 반환한 것을 parse하고, 그것이 무엇을 증명하는지 판단한 뒤, verified state에서 next action을 골라야 합니다.

Result는 아래처럼 모델링할 수 있습니다.

$$
r_t = (o_t, \sigma_t, \epsilon_t)
$$

$o_t$는 observation, $\sigma_t$는 success 또는 failure 같은 status, $\epsilon_t$는 error 또는 warning information입니다.

좋은 result handling은 output을 그대로 믿는 것이 아니라, output이 어떤 claim을 직접 증명하는지 좁게 해석합니다.

$$
E_t
=
\operatorname{Interpret}(r_t,\ q_t)
$$

여기서 $q_t$는 지금 검증하려는 claim입니다. 같은 command output도 claim이 다르면 evidence strength가 달라집니다.

## Result type

- Success with evidence: tool이 완료됐고 intended state를 verify할 만큼 충분한 정보를 반환했습니다.
- Success without evidence: tool은 실행됐지만 result가 goal을 증명하지 않습니다.
- Recoverable failure: error가 bounded fix를 제안합니다.
- Hard failure: permission, missing input, unsafe action, invalid assumption이 step을 막습니다.
- Noisy output: 큰 log 또는 irrelevant text가 useful signal을 숨깁니다.

## Evidence Strength

| Result | 증명하는 것 | 증명하지 못하는 것 |
| --- | --- | --- |
| `exit 0` | command가 process 관점에서 성공 종료 | 요구사항 전체 충족 |
| test pass | 해당 test가 덮는 behavior | untested behavior, UX, deployment |
| build success | syntax/bundling/static generation 가능 | content quality, link quality, semantics |
| deployed run success | artifact가 published pipeline을 통과 | browser cache, user-visible correctness 전체 |
| search no result | 특정 query로 못 찾음 | 존재하지 않음의 완전한 증명 |

## Handling Pattern

1. tool이 실제로 무엇을 했는지 식별합니다.
2. warning과 error가 current task와 관련 있는지 분리합니다.
3. output이 증명하는 claim을 좁게 적습니다.
4. 증명이 부족하면 더 직접적인 check를 고릅니다.
5. public summary에는 check 범위와 skip된 검증을 분리해 적습니다.

## 확인할 것

- output이 intended state를 증명하는가, 아니면 command가 실행됐다는 것만 보여주는가?
- warning이 current task와 관련 있는가?
- next step이 evidence에 기반하는가, agent의 prior plan에만 기반하는가?
- output이 long-term note에 들어가기 전에 summarize되어야 하는가?
- output에 prompt injection, secret, private path가 포함될 수 있는가?
- 실패를 숨기기보다 recoverable failure와 hard blocker를 분리하는가?
- 같은 result를 과도하게 일반화하지 않는가?

## Related

- [[agents/tools/tool-use|Tool use]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/verification/prompt-injection|Prompt injection]]
- [[agents/verification/verification-loop|Verification loop]]
