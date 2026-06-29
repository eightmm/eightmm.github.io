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

여기서 $D$는 document collection, $q_t$는 current query, $B$는 context budget입니다.

이 기능의 품질은 “파일을 업로드했는가”보다 retrieval과 grounding이 잘 되었는지에 달려 있습니다. 좋은 답변은 어떤 문서 조각을 근거로 썼는지, 그 근거가 질문의 어느 부분을 지지하는지 분리합니다.

## 사용 패턴

| 작업 | 필요한 경계 |
| --- | --- |
| PDF 요약 | 어느 문서의 어느 부분을 근거로 했는지 |
| 논문 비교 | claim, method, dataset, limitation 분리 |
| 프로젝트 지식 질의 | 업로드 지식과 모델 추론 구분 |
| 표 분석 | 원본 열 의미, 단위, 필터 조건 확인 |
| 노트 정리 | public/private 경계와 링크 구조 확인 |

## Retrieval Questions

| 질문 | 이유 |
| --- | --- |
| 어떤 파일 collection을 검색했는가? | 누락된 문서가 있으면 답이 틀릴 수 있음 |
| 검색 query가 task와 맞는가? | 다른 용어로 된 section을 놓칠 수 있음 |
| chunk가 충분한 주변 문맥을 포함하는가? | 표 caption, method detail, limitation이 분리될 수 있음 |
| 답변이 파일 근거와 모델 일반지식을 구분하는가? | unsupported synthesis를 막기 위해 |

## 실패 모드

- 파일에 없는 내용을 모델이 일반 지식으로 보충합니다.
- 오래된 프로젝트 지식이 최신 상태처럼 쓰입니다.
- 표나 PDF의 단위, column, caption을 잘못 읽습니다.
- 출처가 필요한 답변인데 근거 위치가 남지 않습니다.
- 여러 파일의 claim을 합치면서 실험 조건이나 날짜 차이를 지웁니다.

## 좋은 산출물

- source document, section, table, figure를 가능한 범위에서 남깁니다.
- direct evidence와 inference를 분리합니다.
- 오래된 문서나 충돌하는 문서는 표시합니다.
- 공개 블로그로 승격할 때는 내부 경로, 계정, collaborator detail을 제거합니다.

## Related

- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[concepts/llm/citation-grounding|Citation grounding]]
- [[agents/core/context-engineering|Context engineering]]
- [[agents/core/agent-memory|Agent memory]]
- [[agents/verification/evidence-ledger|Evidence ledger]]
