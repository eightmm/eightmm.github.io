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

## Result type

- Success with evidence: tool이 완료됐고 intended state를 verify할 만큼 충분한 정보를 반환했습니다.
- Success without evidence: tool은 실행됐지만 result가 goal을 증명하지 않습니다.
- Recoverable failure: error가 bounded fix를 제안합니다.
- Hard failure: permission, missing input, unsafe action, invalid assumption이 step을 막습니다.
- Noisy output: 큰 log 또는 irrelevant text가 useful signal을 숨깁니다.

## 확인할 것

- output이 intended state를 증명하는가, 아니면 command가 실행됐다는 것만 보여주는가?
- warning이 current task와 관련 있는가?
- next step이 evidence에 기반하는가, agent의 prior plan에만 기반하는가?
- output이 long-term note에 들어가기 전에 summarize되어야 하는가?
- output에 prompt injection, secret, private path가 포함될 수 있는가?

## Related

- [[agents/tools/tool-use|Tool use]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/verification/prompt-injection|Prompt injection]]
- [[agents/verification/verification-loop|Verification loop]]
