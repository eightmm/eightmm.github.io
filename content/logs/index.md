---
title: Public Logs
tags:
  - logs
  - public-notes
---

# Public Logs

Log는 정리된 work record입니다. Private operational detail이 아니라 무엇을 배웠는지를 남겨야 합니다.

Public log는 post보다 짧고, project보다 artifact 중심이 약하며, infra runbook보다 사건/결정의 맥락이 강합니다.

$$
\text{event}
\rightarrow
\text{evidence}
\rightarrow
\text{decision}
\rightarrow
\text{public lesson}
$$

## 용도

- 공개 가능한 debugging writeup.
- 결과 공유가 안전해진 뒤의 reproducible experiment summary.
- project milestone과 design decision.
- 나중에 [[posts/index|Posts]]로 확장될 수 있는 짧은 note.

## 어디에 둘까

| 중심 | 둘 곳 |
| --- | --- |
| 짧은 공개 작업 기록, 사건, 결정 | [Public Logs](/logs) |
| 실제 구현물, pipeline, tool, release | [Projects](/projects) |
| 일반화된 서버/HPC 운영 지식 | [Infra](/infra) |
| 독자를 위한 완성된 설명 글 | [Posts](/posts) |
| 논문 하나의 claim과 evidence | [Papers](/papers) |
| 재사용 가능한 정의, 수식, protocol | [Concepts](/concepts) |

## 형식

- [[logs/public-log-taxonomy|Public log taxonomy]]
- [[logs/public-log-format|Public log format]]
- [[logs/log-promotion-rule|Log promotion rule]]
- [[logs/public-incident-note|Public incident note]]
- [[logs/sanitization-checklist|Sanitization checklist]]

## 공개 로그 체크

| Check | Why |
| --- | --- |
| What happened? | event를 일반화된 형태로 남깁니다 |
| What evidence is safe? | raw internal log 대신 error class와 public-safe signal만 씁니다 |
| What decision changed? | 다음에 같은 상황에서 판단 기준이 됩니다 |
| What should be omitted? | private path, host, account, collaborator, unpublished metric을 제거합니다 |
| Where should it promote? | Project, Infra, Post, Paper, Concept 중 다음 위치를 정합니다 |

## 포함하지 말 것

- server IP, port, username, private path, credential, node name.
- internal project name, collaborator detail, private dataset.
- unpublished experiment result 또는 thesis-sensitive claim.

## Related

- [[projects/index|Projects]]
- [[infra/index|Infra]]
- [[research/index|Research]]
- [[inbox/inbox-triage|Inbox triage]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
- [[concepts/systems/failure-recovery|Failure recovery]]
