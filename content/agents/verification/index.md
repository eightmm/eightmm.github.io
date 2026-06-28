---
title: Agent Verification
tags:
  - agents
  - verification
---

# Agent Verification

Agent verification은 agent가 만든 답변, 파일 수정, 보고서, PR, 요약이 실제로 요구사항을 만족하는지 확인하는 절차입니다. Agent가 “끝났다”고 말하는 것과 끝났다는 evidence가 있는 것은 다릅니다.

$$
\operatorname{verified}(c)
=
\exists E\ \text{such that}\ E \Rightarrow c
$$

where $c$ is the claim and $E$ is evidence from tests, builds, rendered pages, logs, source inspection, citations, review, or human judgment. 넓은 claim에는 넓은 evidence가 필요하고, 좁은 check는 그 check가 덮는 범위만 증명합니다.

## Verification Ladder

1. 결과를 판단하기 전에 [[agents/verification/acceptance-criteria|Acceptance criteria]]를 정합니다.
2. [[agents/verification/evidence-ledger|Evidence ledger]]에 무엇을 확인했는지 남깁니다.
3. 의미 있는 side effect 뒤에는 [[agents/verification/verification-loop|Verification loop]]를 실행합니다.
4. [[agents/verification/reflection-and-critique|Reflection and critique]]로 빠진 failure mode를 찾습니다.
5. 넓은 목표를 완료했다고 말하기 전에는 [[agents/verification/completion-audit|Completion audit]]를 수행합니다.

## Notes

- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/acceptance-criteria|Acceptance criteria]]
- [[agents/verification/evidence-ledger|Evidence ledger]]
- [[agents/verification/completion-audit|Completion audit]]
- [[agents/verification/reflection-and-critique|Reflection and critique]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[agents/verification/prompt-injection|Prompt injection]]

## Checks

- 정확히 어떤 claim을 검증하는가?
- evidence가 직접적이고 최신이며 claim 범위와 맞는가?
- 각 check가 무엇을 증명했고, 무엇을 증명하지 못했는가?
- skipped, impossible, too narrow check가 있는가?
- 출력이 private infrastructure, collaborator, credential, unpublished result를 노출하는가?
- final summary가 verified fact와 assumption을 분리하는가?

## Related

- [[agents/index|Agents]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/tools/tool-result-handling|Tool result handling]]
- [[agents/core/agent-operating-contract|Agent operating contract]]
- [[concepts/llm/hallucination-grounding|Hallucination and grounding]]
- [[logs/sanitization-checklist|Sanitization checklist]]
