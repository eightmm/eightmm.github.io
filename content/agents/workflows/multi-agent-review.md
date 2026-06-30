---
title: Multi-Agent Review
tags:
  - agents
  - llm
  - review
---

# Multi-Agent Review

Multi-agent review는 변경을 받아들이기 전에 둘 이상의 agent, 보통 서로 다른 model이나 role이 독립적으로 검토하는 절차입니다. 독립적인 pass는 한 agent가 합리화하고 지나칠 수 있는 error를 잡아냅니다.

가치는 conditional independence에서 나옵니다. Reviewer가 같은 hidden assumption을 공유하면 agreement는 약한 evidence입니다.

$$
P(\text{correct}\mid r_1,r_2)
\not\approx
P(\text{correct}\mid r_1)P(\text{correct}\mid r_2)
$$

Reviewer가 artifact를 독립적으로 inspect하고 서로 다른 failure hypothesis를 사용했을 때만 더 강한 근거가 됩니다.

## 유용한 role

- Implementer: 가장 작고 scope가 맞는 change를 만듭니다.
- Reviewer: bug, missing check, regression을 찾습니다.
- Domain critic: scientific, data, security assumption을 확인합니다.
- Verifier: acceptance check를 실행하고 evidence를 기록합니다.
- Editor: rough output을 public documentation으로 다듬습니다.

## Review Contract

Multi-agent review는 의견 수집이 아니라 검증 가능한 finding을 모으는 절차입니다.

| Field | Required content |
| --- | --- |
| goal | 무엇을 만족해야 하는가 |
| artifact | diff, note, command output, paper claim, workflow |
| reviewer lens | bug, domain, security, writing, reproducibility |
| finding | file, line, claim, or step reference |
| severity | blocking, important, minor, suggestion |
| evidence | why the finding is true |
| disposition | fixed, accepted risk, rejected with reason |

이 contract가 없으면 reviewer agreement가 그냥 취향 투표가 됩니다.

## Independence

독립성은 prompt만 다르게 준다고 생기지 않습니다.

| Shared factor | Risk |
| --- | --- |
| same summary only | all reviewers inherit missing detail |
| same generated draft | all reviewers rationalize the same mistake |
| same hidden assumption | consensus becomes weak evidence |
| no source access | review becomes plausibility checking |
| no verifier | findings cannot be closed |

Strong review는 reviewer가 artifact와 source state를 직접 볼 수 있을 때 나옵니다.

## 실전 check

- Reviewer에게 detail을 숨기는 summary가 아니라 diff와 goal을 줍니다.
- confirmation보다 adversarial prompt, 예를 들어 “이것을 반박해보라”를 선호합니다.
- 독립 model 간 agreement는 proof가 아니라 weak evidence로 취급합니다.
- API, auth, data pipeline, training, dependency 같은 high-risk surface는 review 뒤에 둡니다.
- 나중에 decision을 재검토할 수 있도록 verdict를 기록합니다.
- reviewer consensus를 test 실행이나 source inspection의 대체물로 취급하지 않습니다.
- private data는 prompt와 review artifact에 넣지 않습니다.
- finding마다 accepted, fixed, or rejected status를 남깁니다.
- final owner가 reviewer 의견과 verification evidence를 분리해 판단합니다.

## Public Wiki에서의 사용

Research blog에서는 multi-agent review가 아래 작업에 유용합니다.

- public post sanitization.
- paper-note claim을 source metadata와 대조.
- broken link나 missing definition 탐색.
- publication 전 agent-generated Markdown 검토.
- raw inbox note와 curated wiki page 분리.

## Failure Modes

| Failure | Symptom |
| --- | --- |
| review theater | many comments, no changed acceptance evidence |
| consensus fallacy | several agents agree because they saw the same weak summary |
| unbounded debate | no owner resolves conflicting findings |
| hidden private context | review cannot be reproduced publicly |
| no regression check | prose improves but links/build break |

## Related

- [[agents/core/agent-operating-contract|Agent operating contract]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/core/planning|Planning]]
- [[agents/workflows/coding-agents|Coding agents]]
- [[agents/index|Agents]]
