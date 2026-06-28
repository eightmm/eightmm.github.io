---
title: Files and Knowledge
tags:
  - agents
  - knowledge
  - files
---

# Files and Knowledge

파일 기반 agent 기능은 사용자가 업로드한 PDF, 문서, 표, 노트, 프로젝트 지식을 읽고 답변이나 산출물을 만드는 기능입니다. 핵심은 모델이 모든 파일을 그대로 기억하는 것이 아니라, 작업에 필요한 부분을 선택해 context로 가져오는 것입니다.

$$
C_t = \operatorname{select}(D, q_t, B)
$$

where $D$ is the document collection, $q_t$ is the current query, and $B$ is the context budget.

## 사용 패턴

| 작업 | 필요한 경계 |
| --- | --- |
| PDF 요약 | 어느 문서의 어느 부분을 근거로 했는지 |
| 논문 비교 | claim, method, dataset, limitation 분리 |
| 프로젝트 지식 질의 | 업로드 지식과 모델 추론 구분 |
| 표 분석 | 원본 열 의미, 단위, 필터 조건 확인 |
| 노트 정리 | public/private 경계와 링크 구조 확인 |

## 실패 모드

- 파일에 없는 내용을 모델이 일반 지식으로 보충합니다.
- 오래된 프로젝트 지식이 최신 상태처럼 쓰입니다.
- 표나 PDF의 단위, column, caption을 잘못 읽습니다.
- 출처가 필요한 답변인데 근거 위치가 남지 않습니다.

## Related

- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[concepts/llm/citation-grounding|Citation grounding]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/agent-memory|Agent memory]]
- [[agents/verification/evidence-ledger|Evidence ledger]]
