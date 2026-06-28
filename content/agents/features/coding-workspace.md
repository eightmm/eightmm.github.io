---
title: Coding Workspace
tags:
  - agents
  - coding
---

# Coding Workspace

Coding workspace 기능은 agent가 저장소를 읽고, 파일을 수정하고, 테스트를 실행하고, diff를 설명하는 작업면입니다. GitHub Copilot coding agent, Claude Code, Codex 같은 도구는 이 범주에 들어갑니다.

$$
\text{repo state}
\rightarrow
\text{inspect}
\rightarrow
\text{edit}
\rightarrow
\text{test}
\rightarrow
\text{diff}
\rightarrow
\text{review}
$$

## 일반 기능

| 기능 | 의미 |
| --- | --- |
| Codebase search | 파일, symbol, 테스트, 설정 탐색 |
| Patch editing | 작은 diff를 만들고 기존 스타일 유지 |
| Terminal execution | 테스트, lint, build, script 실행 |
| PR or branch workflow | 변경을 branch/PR 단위로 검토 |
| Review assistance | 버그, regression, missing test 찾기 |

## 좋은 요청

- 이 파일의 버그를 찾아 테스트와 함께 고쳐줘.
- 이 함수의 public behavior를 유지하면서 중복을 줄여줘.
- 이 문서를 현재 코드 기준으로 업데이트해줘.
- 실패한 CI 로그를 보고 최소 수정안을 만들어줘.

## 위험한 요청

- 요구사항이 불명확한 대형 refactor.
- schema, auth, dependency, infrastructure 변경.
- secret, credential, private data가 포함된 작업.
- 검증 경로가 없는 “알아서 개선” 요청.

## Official References

- [GitHub Copilot features](https://docs.github.com/en/copilot/get-started/features)
- [Claude Code overview](https://docs.anthropic.com/en/docs/claude-code/overview)

## Related

- [[agents/workflows/coding-agents|Coding agents]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/verification/acceptance-criteria|Acceptance criteria]]
- [[agents/verification/completion-audit|Completion audit]]
