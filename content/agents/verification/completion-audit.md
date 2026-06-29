---
title: Completion Audit
tags:
  - agents
  - verification
  - workflows
---

# Completion Audit

Completion audit는 agent가 “done”이라고 말해도 되는지 확인합니다. 단순히 test를 실행하는 것보다 엄격하며, full objective를 current evidence와 비교합니다.

Requirement $\mathcal{R}$와 evidence ledger $L$이 있을 때 completion은 아래를 요구합니다.

$$
\operatorname{complete}
=
\bigwedge_{r \in \mathcal{R}}
\operatorname{covered}(r, L)
$$

어떤 requirement라도 unverified, contradicted, evidence에 비해 too broad, missing이면 task는 complete가 아닙니다.

## Audit step

1. 실제 objective를 completed work에 맞춰 좁히지 않고 다시 적습니다.
2. explicit deliverable, command, invariant, safety rule, reporting requirement를 나열합니다.
3. 각 requirement를 증명할 evidence가 무엇인지 적습니다.
4. current file, command output, rendered artifact, external state를 inspect합니다.
5. 각 requirement를 `proved`, `contradicted`, `incomplete`, `weak evidence`, `missing` 중 하나로 표시합니다.
6. 모든 required item이 proved가 아니면 계속 작업합니다.

## 흔한 failure mode

- build 통과를 content correctness의 proof로 취급합니다.
- old memory를 current evidence로 취급합니다.
- skipped check를 passed처럼 보고합니다.
- user가 요청한 것보다 작은 task를 완료합니다.
- public-safety 또는 privacy requirement를 무시합니다.
- workflow가 요구하는데 repository change 뒤 push를 잊습니다.

## 확인할 것

- audit가 original scope를 보존했는가?
- 모든 requirement에 direct evidence가 있는가?
- broad claim이 broad check로 support되는가?
- uncertain item을 명시적으로 not verified로 표시했는가?
- final answer가 newest user request와 일치하는가?

## Related

- [[agents/verification/evidence-ledger|Evidence ledger]]
- [[agents/verification/acceptance-criteria|Acceptance criteria]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/core/agent-operating-contract|Agent operating contract]]
- [[agents/workflows/agent-runbook|Agent runbook]]
