---
title: Reflection and Critique
tags:
  - agents
  - verification
  - workflows
---

# Reflection and Critique

Reflection and critique는 계속 진행하기 전에 agent가 plan, output, failure를 inspect하게 하는 단계입니다. 품질을 높일 수 있지만, evidence와 concrete check에 묶여 있을 때만 효과가 있습니다.

Critique step은 아래처럼 모델링할 수 있습니다.

$$
c_t = \operatorname{Critique}(a_t, E_t, R)
$$

여기서 $a_t$는 proposed action 또는 artifact, $E_t$는 evidence, $R$은 review rubric입니다.

## 유용한 critique 대상

- missing requirement.
- broken link, test, build step.
- unsupported claim.
- risky side effect.
- inconsistent terminology.
- public/private information leakage.

## Failure mode

- critique가 generic advice가 됩니다.
- agent가 자기 output을 rationalize합니다.
- critique가 factual error를 놓치고 style만 봅니다.
- 새 evidence 없이 loop가 계속 revise합니다.

## 확인할 것

- critique가 review 대상 artifact를 cite하는가?
- actionable change를 만들어내는가?
- critique 뒤에 external verification step이 있는가?
- independent reviewer가 다른 error를 잡을 수 있는가?
- 새 evidence가 추가되지 않을 때 workflow가 멈추는가?

## Related

- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[concepts/evaluation/error-analysis|Error analysis]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
