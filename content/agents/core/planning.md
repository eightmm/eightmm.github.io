---
title: Planning
tags:
  - agents
  - llm
  - planning
---

# Planning

Planning은 agent가 행동하기 전에 goal을 순서 있는 verifiable subtask로 분해하는 단계입니다. 좋은 plan은 blast radius를 제한하고, assumption을 드러내며, progress를 checkable하게 만듭니다.

유용한 plan은 각 step을 evidence에 연결합니다.

$$
P = \{(s_i, v_i)\}_{i=1}^{k}
$$

여기서 $s_i$는 step이고 $v_i$는 그것을 verify할 수 있는 check입니다. Verification path가 없는 step은 보통 plan이 아니라 guess입니다.

## Plan 형태

- 현재 evidence를 적습니다.
- 틀릴 수 있는 assumption을 이름 붙입니다.
- 작은 next action을 고릅니다.
- 그 action에 verification check를 붙입니다.
- 새 evidence가 문제를 바꾸면 re-plan합니다.

## 실전 check

- 첫 action 전에 goal, constraint, success criteria를 적습니다.
- 각 step이 concrete verification을 갖도록 작업을 나눕니다.
- step이 실패하거나 새 정보가 assumption과 충돌하면 re-plan합니다.
- original goal에서 drift가 생기는지 보이도록 plan을 유지합니다.
- intent가 ambiguous하면 guess하지 말고 멈춰서 묻습니다.
- plan이 file inspection, check 실행, artifact 변경의 대체물이 되지 않게 합니다.

## Related

- [[agents/core/agent-operating-contract|Agent operating contract]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/core/task-decomposition|Task decomposition]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/tools/tool-use|Tool use]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[agents/index|Agents]]
