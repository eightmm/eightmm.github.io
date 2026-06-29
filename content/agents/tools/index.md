---
title: Agent Tools
tags:
  - agents
  - tools
---

# Agent Tools

Agent tool은 모델의 언어적 결정을 실제 외부 행동으로 바꾸는 인터페이스입니다. 검색, 파일 읽기, 파일 수정, 코드 실행, API 호출, 브라우저 조작, PR 생성처럼 외부 상태를 읽거나 바꾸는 기능이 여기에 들어갑니다.

$$
\text{tool}: (I, S_{\mathrm{pre}}) \rightarrow (O, S_{\mathrm{post}})
$$

여기서 $I$는 입력 payload, $O$는 반환 output, $S_{\mathrm{pre}}, S_{\mathrm{post}}$는 호출 전후의 외부 상태입니다.

Tool을 붙이면 agent는 더 강해지지만, 동시에 더 위험해집니다. 답변 오류는 text 안에 머물 수 있지만 tool 오류는 file, repo, browser, API, public page 같은 외부 상태를 바꿀 수 있습니다.

## Tool 사용 흐름

1. 질문에 답할 수 있는 가장 약한 tool을 고릅니다.
2. [[agents/tools/tool-contract|Tool contract]]에서 schema, side effect, permission, failure mode를 확인합니다.
3. tool output을 새 instruction이 아니라 evidence로 처리합니다.
4. side effect가 있으면 필요한 verification을 실행합니다.
5. 공개 노트, 로그, 요약에는 public-safe output만 남깁니다.

## Tool Risk Map

| Tool type | 예 | 주요 위험 | 필요한 verifier |
| --- | --- | --- | --- |
| Retrieval | search, file read, vector DB | stale or irrelevant evidence | source inspection, citation |
| Computation | calculator, parser, script | wrong input, silent failure | output sanity check |
| File write | patch, formatter, generator | unintended diff | git diff, build/test |
| External write | issue, PR, message, email | public or irreversible side effect | preview, confirmation, remote status |
| Browser/computer | click, type, submit | stateful mutation | screenshot/log, human gate |

## Notes

- [[agents/tools/tool-use|Tool use]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/tools/tool-result-handling|Tool result handling]]

## Checks

- 이 tool은 read-only인가, side-effecting인가?
- 입력에 secret, private path, hostname, username, unpublished result가 들어가는가?
- 출력이 검증 가능한 구조인가?
- retry가 안전한가, 중복 side effect를 만들 수 있는가?
- side effect가 올바르게 일어났다는 evidence는 무엇인가?
- tool이 실패했을 때 natural-language fallback으로 성공처럼 보이지 않는가?

## Related

- [[agents/index|Agents]]
- [[agents/features/connectors-actions|Connectors and actions]]
- [[agents/core/action-space|Action space]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/verification/evidence-ledger|Evidence ledger]]
- [[concepts/llm/structured-output|Structured output]]
- [[agents/verification/verification-loop|Verification loop]]
