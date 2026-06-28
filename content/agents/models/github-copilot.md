---
title: GitHub Copilot
tags:
  - agents
  - models
  - coding
---

# GitHub Copilot

GitHub Copilot은 IDE와 GitHub workflow에 붙은 coding assistant family입니다. completion, chat, code review, cloud coding agent, PR workflow 같은 기능이 codebase-centered agent 경험을 만듭니다.

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

## Official References

- [GitHub Copilot features](https://docs.github.com/en/copilot/get-started/features)
- [GitHub Copilot product page](https://github.com/features/copilot)

## Related

- [[agents/features/coding-workspace|Coding workspace]]
- [[agents/workflows/coding-agents|Coding agents]]
- [[agents/verification/completion-audit|Completion audit]]
- [[agents/models/index|Agent model families]]
