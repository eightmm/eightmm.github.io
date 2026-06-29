---
title: Agent Evaluation
tags:
  - agents
  - evaluation
  - workflows
---

# Agent Evaluation

Agent evaluation은 model이 그럴듯하게 말하는지가 아니라 agent workflow가 correct artifact를 안정적으로 만드는지 측정합니다. Evaluation unit은 완료된 task와 evidence입니다.

간단한 task success estimate는 아래와 같습니다.

$$
\hat{p}_{\mathrm{success}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[\operatorname{pass}(T_i)=1]
$$

여기서 $T_i$는 task이고 $\operatorname{pass}$는 외부에서 확인한 success condition입니다.

이 estimate는 uncertainty와 함께 보고해야 합니다. Bernoulli pass rate의 rough standard error는 아래와 같습니다.

$$
\operatorname{SE}(\hat{p})
=
\sqrt{
\frac{\hat{p}(1-\hat{p})}{N}
}
$$

작은 agent eval set은 fragile workflow를 안정적으로 보이게 만들 수 있으므로 이 값이 중요합니다.

## 측정할 것

- 명확한 rubric 아래의 task success.
- tool use와 file edit의 correctness.
- verification coverage.
- security와 privacy violation.
- cost, latency, retry 횟수.
- human review burden.

## Evaluation unit

Agent task에는 보통 아래 항목이 포함되어야 합니다.

- initial state.
- user request.
- available tools and permissions.
- expected artifact.
- acceptance criteria.
- verification evidence.
- 통과하지 못했을 때의 failure label.

Coding 또는 wiki workflow에서는 보통 artifact quality와 verification trail이 모두 있어야 pass로 봅니다.

$$
\operatorname{pass}(T)
=
\operatorname{artifact\_ok}(T)
\land
\operatorname{evidence\_ok}(T)
\land
\operatorname{safety\_ok}(T)
$$

## Failure taxonomy

- Planning failure: decomposition이 틀렸거나, 순서가 틀렸거나, 너무 일찍 끝낸 경우.
- Context failure: 관련 file을 놓치거나, stale memory를 쓰거나, hallucinated constraint를 만든 경우.
- Tool failure: command, path, side effect가 틀렸거나 error를 무시한 경우.
- Domain failure: 그럴듯하지만 기술적으로 틀린 content를 만든 경우.
- Verification failure: check가 부족하거나 claim이 과도하게 넓은 경우.
- Safety failure: secret exposure, prompt-injection obedience, unsafe publication.
- Handoff failure: 다른 agent나 사람이 일어난 일을 재현할 수 없는 경우.

## Coverage

Agent evaluation은 clean demo뿐 아니라 현실적인 variation을 포함해야 합니다.

$$
\mathcal{T}_{\mathrm{eval}}
=
\mathcal{T}_{\mathrm{happy}}
\cup
\mathcal{T}_{\mathrm{messy}}
\cup
\mathcal{T}_{\mathrm{adversarial}}
\cup
\mathcal{T}_{\mathrm{regression}}
$$

Messy task에는 dirty worktree, ambiguous instruction, flaky tool, partial prior work, outdated doc이 포함됩니다. Adversarial task에는 prompt injection, malicious file, misleading tool output이 포함됩니다.

## 확인할 것

- success를 test, review, build output, external ground truth 중 무엇으로 판단하는가?
- failure를 planning, tool use, context, verification, domain knowledge 기준으로 분류하는가?
- benchmark가 현실적인 messy state를 포함하는가?
- private data와 credential이 evaluation trace에서 제외되는가?
- evaluation이 model quality를 tool permission, scaffold quality, verifier quality와 분리하는가?
- saved prompt, input, state, artifact로 task outcome을 재현할 수 있는가?
- model, prompt, tool, policy 변경에 따른 regression을 추적하는가?

## Related

- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/acceptance-criteria|Acceptance criteria]]
- [[agents/verification/evidence-ledger|Evidence ledger]]
- [[agents/verification/completion-audit|Completion audit]]
- [[agents/verification/prompt-injection|Prompt injection]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[concepts/evaluation/metric|Metric]]
- [[papers/analysis/benchmark-card|Benchmark card]]
- [[concepts/evaluation/evaluation-set-design|Evaluation set design]]
- [[concepts/math/statistical-estimator|Statistical estimator]]
- [[concepts/learning/reward-modeling|Reward modeling]]
- [[concepts/learning/preference-optimization|Preference optimization]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
