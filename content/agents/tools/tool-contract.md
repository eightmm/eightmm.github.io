---
title: Tool Contract
tags:
  - agents
  - tools
  - verification
---

# Tool Contract

Tool contract는 tool이 무엇을 할 수 있고, 어떤 input을 받으며, output이 무엇을 뜻하고, 어떤 side effect를 만들 수 있는지 정의합니다. 명확한 contract가 없으면 agent tool use는 guesswork가 됩니다.

Contract는 아래처럼 요약할 수 있습니다.

$$
\text{tool}: (I, S_{\mathrm{pre}}) \rightarrow (O, S_{\mathrm{post}})
$$

여기서 $I$는 input, $O$는 output, $S_{\mathrm{pre}}, S_{\mathrm{post}}$는 call 전후의 external state입니다.

Agent에게 tool은 단순한 helper가 아니라 environment를 읽거나 바꾸는 interface입니다. 따라서 tool description에는 “무엇을 할 수 있다”뿐 아니라 “무엇을 증명하지 못한다”도 포함되어야 합니다.

## Contract element

- input schema와 required field.
- output schema와 error format.
- side effect: file edit, network call, process launch, state change, write.
- precondition과 permission boundary.
- successful execution 뒤의 verification path.
- failure behavior와 retry policy.

## Tool Class

| Class | 예 | 기본 검증 |
| --- | --- | --- |
| Read-only | file read, search, log inspect | output이 최신 source에서 왔는지 확인 |
| Transform | formatter, parser, summarizer | diff 또는 transformed artifact 확인 |
| Side-effecting | file edit, API write, deploy, push | external state가 의도대로 바뀌었는지 확인 |
| Long-running | build, training, queue job | run id, log, exit status, artifact 확인 |
| Human-facing | notification, PR, public page | rendered output과 audience risk 확인 |

## Minimal Contract Template

| Field | 질문 |
| --- | --- |
| Purpose | 이 tool은 어떤 decision에 필요한 evidence를 만드는가? |
| Input | required field와 invalid input은 무엇인가? |
| Output | success, warning, failure를 어떻게 구분하는가? |
| Side effect | 어떤 file, external state, public artifact를 바꾸는가? |
| Permission | 언제 approval이 필요한가? |
| Verification | 실행 뒤 어떤 check가 success를 증명하는가? |

## 확인할 것

- tool output을 instruction이 아니라 data로 취급하는가?
- tool이 public artifact나 external state를 바꿀 수 있는가?
- output만으로 success를 verify하기에 충분한가?
- secret이나 private path가 log에서 제외되는가?
- inspection에 더 안전한 read-only tool이 있는가?
- retry가 같은 side effect를 중복 생성하지 않는가?
- failure output이 다음 행동을 정할 만큼 structured한가?

## Related

- [[agents/tools/tool-use|Tool use]]
- [[concepts/systems/inference-contract|Inference contract]]
- [[agents/verification/prompt-injection|Prompt injection]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[concepts/systems/model-serving|Model serving]]
