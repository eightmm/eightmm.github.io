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

여기서 $c$는 claim이고 $E$는 test, build, rendered page, log, source inspection, citation, review, human judgment에서 나온 evidence입니다. 넓은 claim에는 넓은 evidence가 필요하고, 좁은 check는 그 check가 덮는 범위만 증명합니다.

## Verification 단계

1. 결과를 판단하기 전에 [[agents/verification/acceptance-criteria|Acceptance criteria]]를 정합니다.
2. [[agents/verification/evidence-ledger|Evidence ledger]]에 무엇을 확인했는지 남깁니다.
3. 의미 있는 side effect 뒤에는 [[agents/verification/verification-loop|Verification loop]]를 실행합니다.
4. [[agents/verification/reflection-and-critique|Reflection and critique]]로 빠진 failure mode를 찾습니다.
5. 넓은 목표를 완료했다고 말하기 전에는 [[agents/verification/completion-audit|Completion audit]]를 수행합니다.

## Claim Scope

Verification은 claim의 범위를 먼저 정해야 합니다. 같은 evidence라도 좁은 claim에는 충분하고 넓은 claim에는 부족할 수 있습니다.

| Claim | Useful evidence | Not enough |
| --- | --- | --- |
| Markdown syntax is valid | static build or parser result | content is correct |
| Wikilinks resolve | link checker over content tree | rendered UX is good |
| Code compiles | build or typecheck | behavior meets user need |
| Deployment succeeded | remote run status and published artifact | every page is visually correct |
| Summary is faithful | source inspection and cited lines | model memory of prior text |
| Public note is safe | sanitization scan and manual review | absence of obvious secrets in one query |

Verification should state both sides:

$$
E \Rightarrow c_{\mathrm{narrow}}
\quad \not\Rightarrow \quad
c_{\mathrm{broad}}
$$

This prevents `build passed` from silently becoming `the whole task is done`.

## Verification Ladder

Agent 작업은 risk에 따라 verification을 넓힙니다.

| Risk level | Minimum check | Broader check |
| --- | --- | --- |
| Text-only note edit | diff review, link check | site build, rendered page spot check |
| Navigation or sidebar change | build and route check | browser inspection across key pages |
| Code behavior change | focused test | integration or end-to-end test |
| Deployment change | local build | remote workflow and published page |
| Public safety change | secret/sensitive-term scan | manual review of changed pages |

The ladder is not bureaucracy. It matches verification cost to possible blast radius.

## Evidence Ledger Shape

A useful ledger records what was checked and what remains outside the check.

| Field | Write |
| --- | --- |
| Claim | exact statement being verified |
| Evidence | command, file, rendered page, source, review |
| Result | pass, fail, warning, not checked |
| Scope | what the evidence covers |
| Gap | what it does not cover |
| Next action | fix, widen check, or report not verified |

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
- broad objective를 완료했다고 말하기 전에 completion audit가 있는가?

## Related

- [[agents/index|Agents]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/tools/tool-result-handling|Tool result handling]]
- [[agents/core/agent-operating-contract|Agent operating contract]]
- [[concepts/llm/hallucination-grounding|Hallucination and grounding]]
- [[logs/sanitization-checklist|Sanitization checklist]]
