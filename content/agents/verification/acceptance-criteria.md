---
title: Acceptance Criteria
tags:
  - agents
  - verification
---

# Acceptance Criteria

Acceptance criteria는 agent가 task completion을 주장하기 전에 무엇이 true여야 하는지 정의합니다. Vague goal을 checkable condition으로 바꾸는 역할입니다.

Task goal $G$에 대해 criteria를 아래처럼 정의합니다.

$$
\mathcal{C}(G) = \{c_1, c_2, \ldots, c_k\}
$$

Completion은 모든 criterion에 대한 evidence를 요구합니다.

$$
\operatorname{done}(G)
= \bigwedge_{i=1}^{k} \operatorname{verified}(c_i)
$$

Criteria는 task를 시작하기 전에 완벽할 필요는 없지만, completion을 주장하기 전에는 explicit해야 합니다. 작업 중 scope가 바뀌면 criteria도 같이 갱신해야 합니다.

## 좋은 criteria

- Observable: file, build output, test result, rendered page, review decision으로 증명할 수 있습니다.
- Specific: condition이 check할 만큼 좁습니다.
- Complete: correctness, safety, publication constraint를 덮습니다.
- Current: evidence가 이전 memory가 아니라 current environment에서 나옵니다.
- Evidence-linked: 모든 criterion이 [[agents/verification/evidence-ledger|Evidence ledger]] entry 또는 equivalent check로 연결됩니다.
- Public-safe: artifact가 published될 때 privacy와 sanitization을 포함합니다.

## Criteria Template

| Field | 질문 |
| --- | --- |
| Artifact | 무엇이 존재하거나 바뀌어야 하는가? |
| Behavior | 사용자가 어떤 결과를 확인할 수 있어야 하는가? |
| Safety | 공개/권한/비용/데이터 경계는 무엇인가? |
| Verification | 어떤 command, rendered check, review가 증명하는가? |
| Reporting | final answer에 무엇을 밝혀야 하는가? |

## 예시

- expected path에 Markdown page가 존재합니다.
- 모든 internal wikilink가 resolve됩니다.
- `npx quartz build`가 성공합니다.
- generated public page가 private infrastructure detail을 노출하지 않습니다.
- expected branch에 commit이 push되었습니다.

## Bad Criteria

| 나쁜 기준 | 문제 |
| --- | --- |
| “내용을 좋게 만든다” | observable하지 않음 |
| “빌드가 된다” | content quality나 route correctness를 다 덮지 않음 |
| “대충 정리한다” | scope와 stop condition이 없음 |
| “agent가 확인했다” | evidence source가 약함 |

## 확인할 것

- 각 criterion을 증명하는 evidence는 무엇인가?
- test가 claim 범위에 충분히 넓은가?
- skipped verification을 명시적으로 보고하는가?
- generated file을 manual edit에서 제외하는가?
- final answer가 changed, verified, not verified item을 구분하는가?
- commit, push, public-safety requirement가 적용될 때 acceptance set이 그것을 포함하는가?

## Related

- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/evidence-ledger|Evidence ledger]]
- [[agents/verification/completion-audit|Completion audit]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[agents/workflows/agent-runbook|Agent runbook]]
- [[logs/sanitization-checklist|Sanitization checklist]]
