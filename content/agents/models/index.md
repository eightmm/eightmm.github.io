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

제품 비교는 “어느 모델이 제일 좋은가”보다 “내 workflow에서 어떤 interface와 verifier를 제공하는가”를 봐야 합니다. 같은 benchmark 점수라도 file workflow, browser action, coding workspace, memory boundary가 다르면 실제 사용성은 달라집니다.

## 비교 축

| 축 | 볼 것 |
| --- | --- |
| Reasoning | 긴 계획, 복잡한 지시, self-check 능력 |
| Context | 긴 문서, 프로젝트, retrieval 처리 |
| Tools | search, code, files, browser, connectors |
| Workspace | canvas, artifacts, IDE, repo, PR |
| Memory | user memory, project memory, chat history |
| Verification | citations, tests, diff, audit, human review |

## Stable Comparison Questions

| 질문 | 보는 이유 |
| --- | --- |
| context source를 사용자가 확인할 수 있는가? | grounding과 privacy boundary |
| tool action 전 preview/approval이 있는가? | side effect control |
| artifact가 chat과 분리되어 유지되는가? | long-form editing |
| coding task에서 diff/test/branch가 보이는가? | reviewability |
| memory를 project별로 제한할 수 있는가? | stale/private context 방지 |
| 출처와 verification evidence를 남길 수 있는가? | public report quality |

## 제품 노트

| 제품군 | 먼저 볼 노트 | 강한 사용처 |
| --- | --- | --- |
| ChatGPT | [ChatGPT](/agents/models/chatgpt) | general chat, research, apps, canvas, agent tasks |
| Claude | [Claude](/agents/models/claude) | long-form writing, artifacts, coding, tool use |
| Gemini | [Gemini](/agents/models/gemini) | Google ecosystem, Deep Research, multimodal work |
| GitHub Copilot | [GitHub Copilot](/agents/models/github-copilot) | IDE completion, coding agent, PR/repo workflow |

## Reading Rule

제품별 note는 feature inventory가 아니라 agent surface를 읽는 entry point입니다. 최신 기능 여부는 공식 문서에서 확인하고, 이 wiki에서는 context, tool, workspace, memory, verification이라는 stable frame으로 기록합니다.

## Related

- [[agents/features/index|Agent features]]
- [[agents/core/agent-architecture|Agent architecture]]
- [[agents/workflows/coding-agents|Coding agents]]
- [[ai/evaluation|Evaluation]]
