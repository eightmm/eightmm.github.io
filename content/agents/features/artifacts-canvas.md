---
title: Artifacts and Canvas
tags:
  - agents
  - artifacts
  - canvas
---

# Artifacts and Canvas

Artifacts, Canvas, workspace editor 같은 기능은 대화창과 산출물을 분리합니다. 사용자는 긴 문서, 코드, 앱, 표, 시각화를 별도 작업면에서 보고 수정할 수 있습니다.

$$
\text{conversation}
\rightarrow
\text{artifact}
\rightarrow
\text{revision loop}
$$

## 왜 중요한가

| 문제 | 작업면 기능이 해결하는 방식 |
| --- | --- |
| 긴 산출물이 대화에 묻힘 | 별도 pane에서 artifact 유지 |
| 부분 수정이 어려움 | 선택 영역이나 파일 단위로 수정 |
| 코드/문서 반복 수정 | 변경 이력을 좁게 추적 |
| 공유가 필요함 | share, publish, export 기능 사용 |

## 사용 예

- 긴 글 초안 작성과 부분 rewrite.
- 작은 웹 앱이나 UI prototype 생성.
- 데이터 표, chart, 계산 결과 정리.
- 문서 구조를 보면서 계속 수정.

## Official References

- [ChatGPT Canvas](https://help.openai.com/en/articles/9930697-what-is-the-canvas-feature-in-chatgpt)
- [Claude Artifacts](https://support.anthropic.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them)
- [Gemini Canvas](https://support.google.com/gemini/answer/16047321)

## Related

- [[agents/features/chat-and-prompting|Chat and prompting]]
- [[agents/workflows/content-promotion-workflow|Content promotion workflow]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
