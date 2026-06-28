---
title: Agent Workflows
tags:
  - agents
  - workflows
---

# Agent Workflows

Agent workflow는 agent 기능을 반복 가능한 작업 절차로 묶은 것입니다. 여기서는 특정 개인 자동화가 아니라 많은 사용자가 공통으로 쓰는 coding, research, review, handoff, wiki maintenance 같은 패턴을 다룹니다.

$$
\text{input}
\rightarrow
\text{triage}
\rightarrow
\text{plan}
\rightarrow
\text{act}
\rightarrow
\text{verify}
\rightarrow
\text{publish or hand off}
$$

같은 agent architecture라도 workflow마다 acceptance criteria, side-effect boundary, verification ladder가 달라져야 합니다.

## Workflow Families

| Family | 의미 |
| --- | --- |
| Coding | source inspect, narrow edit, test, diff review |
| Paper brief | candidate intake, metadata check, selected note promotion |
| LLM Wiki maintenance | raw input을 concept, paper, project, post로 정리 |
| Multi-agent review | 위험한 변경을 독립적인 관점으로 재검토 |
| Handoff | state, evidence, open decision을 다음 사람이나 agent에게 전달 |

## Notes

- [[agents/workflows/coding-agents|Coding agents]]
- [[agents/workflows/paper-brief-workflow|Paper brief workflow]]
- [[agents/workflows/agent-orchestration|Agent orchestration]]
- [[agents/workflows/agent-handoff|Agent handoff]]
- [[agents/workflows/agent-runbook|Agent runbook]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[agents/workflows/llm-wiki|LLM Wiki]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]

## Checks

- 끝에 어떤 artifact가 존재해야 하는가?
- 허용되는 side effect는 file edit, commit, push, issue, job, publication 중 무엇인가?
- artifact가 충분히 맞다는 evidence는 무엇인가?
- 무엇은 inbox에 남고, 무엇은 wiki note/post/project로 승격되는가?
- public release 전에 사람이 봐야 하는 부분은 무엇인가?

## Related

- [[agents/index|Agents]]
- [[agents/features/index|Agent features]]
- [[agents/verification/index|Agent verification]]
- [[projects/paper-brief-agent-pipeline|Paper brief agent pipeline]]
- [[projects/llm-wiki-blog|LLM Wiki blog]]
- [[inbox/index|Inbox]]
- [[logs/index|Public logs]]
