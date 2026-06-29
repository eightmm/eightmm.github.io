---
title: Agent Model Families
tags:
  - agents
  - models
---

# Agent Model Families

여기서는 유명 LLM 제품을 benchmark 순위가 아니라 agent 기능을 제공하는 사용자-facing interface로 봅니다. 모델명, 가격, plan 제한은 자주 바뀌므로 이 페이지에서는 안정적인 비교 축만 둡니다.

$$
\text{product}
\approx
\text{model family}
+ \text{interface}
+ \text{tools}
+ \text{memory}
+ \text{workspace}
+ \text{policy}
$$

## 비교 축

| 축 | 볼 것 |
| --- | --- |
| Reasoning | 긴 계획, 복잡한 지시, self-check 능력 |
| Context | 긴 문서, 프로젝트, retrieval 처리 |
| Tools | search, code, files, browser, connectors |
| Workspace | canvas, artifacts, IDE, repo, PR |
| Memory | user memory, project memory, chat history |
| Verification | citations, tests, diff, audit, human review |

## 제품 노트

| 제품군 | 먼저 볼 노트 | 강한 사용처 |
| --- | --- | --- |
| ChatGPT | [ChatGPT](/agents/models/chatgpt) | general chat, research, apps, canvas, agent tasks |
| Claude | [Claude](/agents/models/claude) | long-form writing, artifacts, coding, tool use |
| Gemini | [Gemini](/agents/models/gemini) | Google ecosystem, Deep Research, multimodal work |
| GitHub Copilot | [GitHub Copilot](/agents/models/github-copilot) | IDE completion, coding agent, PR/repo workflow |

## Related

- [[agents/features/index|Agent features]]
- [[agents/core/agent-architecture|Agent architecture]]
- [[agents/workflows/coding-agents|Coding agents]]
- [[ai/evaluation|Evaluation]]
