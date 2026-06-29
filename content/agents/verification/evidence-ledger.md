---
title: Evidence Ledger
tags:
  - agents
  - verification
  - evidence
---

# Evidence Ledger

Evidence ledger는 agent가 claim을 만들기 전에 무엇을 확인했는지 기록합니다. Final answer가 memory, intent, 또는 실제 requirement를 덮지 않는 passing check에 기대지 않게 막습니다.

Claim $c$에 대해 ledger는 아래를 저장해야 합니다.

$$
L(c)
=
\{(r_i, e_i, v_i, t_i)\}_{i=1}^{n}
$$

여기서 $r_i$는 requirement, $e_i$는 evidence, $v_i$는 verification method, $t_i$는 check의 time 또는 state입니다.

## Evidence type

| Evidence | 증명하는 것 |
| --- | --- |
| File diff | 무엇이 바뀌었는지 |
| Command output | build, test, lint, scan result |
| Rendered page check | generated output이 link/display되는지 |
| Source citation | fact가 grounded되어 있는지 |
| Review note | human 또는 second-agent judgment |
| Run artifact | config, log, metric, prediction, checkpoint |

## Ledger field

- Requirement: 확인 중인 explicit condition.
- Evidence source: file, command, rendered output, source document, reviewer.
- Coverage: evidence가 무엇을 증명하고 무엇을 증명하지 않는지.
- Result: pass, fail, inconclusive, skipped, blocked.
- Freshness: evidence가 current state에서 왔는지.
- Public boundary: evidence를 quote할 수 있는지, summarize해야 하는지.

## 확인할 것

- 모든 completion claim이 evidence를 가리키는가?
- evidence가 current, direct, broad enough한가?
- skipped 또는 impossible check를 `not verified`로 표시하는가?
- passing build를 content accuracy proof로 과도하게 쓰지 않는가?
- private log, path, hostname, account name, unpublished result를 생략하는가?

## Related

- [[agents/verification/acceptance-criteria|Acceptance criteria]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/completion-audit|Completion audit]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[logs/sanitization-checklist|Sanitization checklist]]
