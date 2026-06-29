---
title: Verification Loop
tags:
  - agents
  - llm
  - verification
---

# Verification Loop

Verification loop는 “agent가 끝났다고 말했다”와 “artifact가 실제로 correct하다” 사이의 gap을 닫습니다. 각 change 뒤에 agent는 concrete check를 실행하고, 그 결과를 다음 decision으로 다시 넣습니다.

이 loop는 claim $c$를 evidence $E$와 비교합니다.

$$
\operatorname{verified}(c)
=
\exists E\; \text{such that}\; E \Rightarrow c
$$

Evidence가 missing, too narrow, stale, indirect라면 claim은 verified가 아닙니다.

Agent가 success를 summarize하기 전에 loop는 [[agents/verification/evidence-ledger|Evidence ledger]]에 evidence를 추가해야 합니다.

## Verification ladder

- syntax 또는 format check.
- 좁은 unit 또는 link check.
- build 또는 integration check.
- runtime smoke test 또는 rendered-page check.
- security, privacy, data leakage, scientific validity review.

## 실전 check

- 모든 change는 build, test, lint, smoke run 같은 real check로 끝냅니다.
- 가장 좁고 유용한 check를 먼저 쓰고, risk가 크면 넓힙니다.
- failing check는 같은 action을 retry하라는 신호가 아니라 fix할 signal로 봅니다.
- skipped 또는 impossible check 위에서 success를 보고하지 않습니다. `not verified`라고 말합니다.
- result를 재현할 수 있도록 command와 output을 남깁니다.
- check를 claim과 맞춥니다. Build 통과가 content accuracy를 증명하지는 않습니다.
- green check는 실제로 cover하는 behavior에 대한 evidence로만 취급합니다.
- broad objective 완료를 주장하기 전에는 [[agents/verification/completion-audit|Completion audit]]를 실행합니다.

## Related

- [[agents/core/agent-operating-contract|Agent operating contract]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/verification/acceptance-criteria|Acceptance criteria]]
- [[agents/verification/evidence-ledger|Evidence ledger]]
- [[agents/verification/completion-audit|Completion audit]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[agents/core/planning|Planning]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[agents/tools/tool-use|Tool use]]
- [[concepts/evaluation/index|Evaluation]]
