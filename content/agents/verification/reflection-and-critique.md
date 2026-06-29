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

Reflection은 모델이 “다시 생각한다”는 뜻이 아니라, output을 requirement와 evidence에 다시 대조하는 step입니다. Critique가 useful하려면 target artifact, rubric, expected fix가 있어야 합니다.

$$
\Delta_t
=
\operatorname{FindGap}(a_t,\ \mathcal{R},\ E_t)
$$

여기서 $\Delta_t$는 missing requirement, weak evidence, contradiction, unsafe output 같은 gap입니다.

## 유용한 critique 대상

- missing requirement.
- broken link, test, build step.
- unsupported claim.
- risky side effect.
- inconsistent terminology.
- public/private information leakage.

## Critique Type

| Type | 질문 | 좋은 결과 |
| --- | --- | --- |
| Requirement critique | 요청한 것이 빠졌는가? | missing item list |
| Evidence critique | claim을 증명하는 check가 있는가? | stronger verification step |
| Safety critique | private/security/detail leakage가 있는가? | remove or generalize |
| Structure critique | 문서 위치와 링크가 맞는가? | route/link fix |
| Technical critique | 수식, code, command가 맞는가? | concrete correction |

## Failure mode

- critique가 generic advice가 됩니다.
- agent가 자기 output을 rationalize합니다.
- critique가 factual error를 놓치고 style만 봅니다.
- 새 evidence 없이 loop가 계속 revise합니다.
- critique가 artifact line, path, claim을 가리키지 않습니다.

## 확인할 것

- critique가 review 대상 artifact를 cite하는가?
- actionable change를 만들어내는가?
- critique 뒤에 external verification step이 있는가?
- independent reviewer가 다른 error를 잡을 수 있는가?
- 새 evidence가 추가되지 않을 때 workflow가 멈추는가?
- critique가 “좋아 보인다”가 아니라 “무엇이 부족하다”를 말하는가?

## Related

- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[concepts/evaluation/error-analysis|Error analysis]]
- [[papers/workflows/paper-review-workflow|Paper review workflow]]
