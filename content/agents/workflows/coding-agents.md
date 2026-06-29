---
title: Coding Agents
tags:
  - agents
  - coding
  - research-engineering
---

# Coding Agents

Coding agent는 codebase를 inspect하고, file을 수정하고, command를 실행하고, 결과를 보고할 수 있는 LLM 기반 tool입니다. Task에 clear goal, local verification path, bounded blast radius가 있을 때 가장 유용합니다.

Coding agent workflow는 아래처럼 보는 편이 좋습니다.

$$
\text{request}
\rightarrow
\text{inspect}
\rightarrow
\text{minimal edit}
\rightarrow
\text{verify}
\rightarrow
\text{diff report}
$$

여기서 핵심은 “코드를 쓸 수 있다”가 아니라 “현재 repository state를 읽고, 변경 범위를 제한하고, 결과를 증명할 수 있다”입니다.

## 잘 맞는 사용처

- test가 있는 작은 module refactor.
- existing code를 바탕으로 documentation draft 작성.
- 알고 있는 codebase에 narrow feature 추가.
- 여러 file에 걸친 repetitive check 실행.
- commit 전 implementation risk review.

## 약한 지점

- vague product direction.
- hidden data 또는 environment assumption.
- explicit review 없는 security-sensitive change.
- written plan 없는 큰 dependency, API, schema, training change.

## Workflow Contract

| 단계 | 해야 할 일 | evidence |
| --- | --- | --- |
| Intake | goal, constraint, no-touch area 확인 | user request, repo instructions |
| Inspect | 관련 파일과 기존 패턴 확인 | source snippets, `rg`, status |
| Edit | 작은 diff로 구현 | staged/unstaged diff |
| Verify | build, test, lint, smoke check 실행 | command output |
| Report | changed, verified, not verified 분리 | final summary |

## Verification 습관

Agent가 도운 모든 change는 build, unit test, lint, smoke test, manual review 같은 concrete check로 끝나야 합니다. 중요한 질문은 agent가 confident하게 말했는지가 아니라 artifact가 correct한지입니다.

## 위험한 작업

아래는 coding agent가 바로 실행하기보다 spec과 approval을 먼저 세워야 합니다.

- public API, database schema, checkpoint format 변경.
- dependency, toolchain, deployment workflow 변경.
- auth, secret, permission, billing, data deletion 관련 작업.
- long-running training, Slurm job, expensive cloud job.
- unpublished research result 또는 private infrastructure가 public artifact로 나갈 수 있는 작업.

## Related

- [[agents/index|Agents]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[projects/index|Projects]]
- [[infra/index|Infra]]
