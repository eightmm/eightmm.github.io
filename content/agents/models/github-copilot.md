---
title: GitHub Copilot
tags:
  - agents
  - models
  - coding
---

# GitHub Copilot

GitHub Copilot은 IDE와 GitHub workflow에 붙은 coding assistant family입니다. completion, chat, code review, cloud coding agent, PR workflow 같은 기능이 codebase-centered agent 경험을 만듭니다.

Copilot은 general assistant라기보다 repository environment에 강하게 붙은 coding surface입니다. 따라서 모델 답변보다 diff, test, PR review, issue scope가 더 중요한 evidence입니다.

## 기능 축

| 기능 | Agent 관점 |
| --- | --- |
| Completion | 현재 파일/커서 context 기반 다음 코드 제안 |
| Chat | 코드 설명, 수정 제안, 질의응답 |
| Code review | diff나 PR의 위험 지점 확인 |
| Coding agent | issue나 task를 받아 branch에서 구현 |
| Model selection | 작업 성격에 따라 모델 선택 |
| IDE/GitHub integration | editor, repo, issue, PR context 활용 |

## 잘 맞는 작업

- 작은 구현 단위.
- 테스트가 있는 bug fix.
- PR review와 설명.
- 반복적인 boilerplate 작성.
- repository-local documentation 정리.

## 확인할 것

- completion은 빠르지만 전체 설계를 보장하지 않습니다.
- coding agent 결과는 PR diff와 test 결과로 검토해야 합니다.
- issue description이 애매하면 agent도 잘못된 범위를 구현하기 쉽습니다.

## Workflow Fit

| Workflow | 확인할 것 |
| --- | --- |
| IDE completion | local context가 충분한가 |
| Chat-based code help | 답이 current codebase와 맞는가 |
| PR review | finding이 file/line/evidence에 묶였는가 |
| Coding agent task | issue scope, branch, test, PR diff가 명확한가 |
| Documentation update | docs가 실제 source와 동기화됐는가 |

## Risk

- completion은 빠르지만 cross-file invariant를 놓칠 수 있습니다.
- coding agent가 생성한 PR도 human review와 CI가 필요합니다.
- repository secret, internal path, private data를 prompt나 issue에 넣지 않아야 합니다.

## Official References

- [GitHub Copilot features](https://docs.github.com/en/copilot/get-started/features)
- [GitHub Copilot product page](https://github.com/features/copilot)

## Related

- [[agents/features/coding-workspace|Coding workspace]]
- [[agents/workflows/coding-agents|Coding agents]]
- [[agents/verification/completion-audit|Completion audit]]
- [[agents/models/index|Agent model families]]
