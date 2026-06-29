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

## Contract element

- input schema와 required field.
- output schema와 error format.
- side effect: file edit, network call, process launch, state change, write.
- precondition과 permission boundary.
- successful execution 뒤의 verification path.
- failure behavior와 retry policy.

## 확인할 것

- tool output을 instruction이 아니라 data로 취급하는가?
- tool이 public artifact나 external state를 바꿀 수 있는가?
- output만으로 success를 verify하기에 충분한가?
- secret이나 private path가 log에서 제외되는가?
- inspection에 더 안전한 read-only tool이 있는가?

## Related

- [[agents/tools/tool-use|Tool use]]
- [[concepts/systems/inference-contract|Inference contract]]
- [[agents/verification/prompt-injection|Prompt injection]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[concepts/systems/model-serving|Model serving]]
