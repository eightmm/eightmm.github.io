---
title: Agent Evaluation
tags:
  - agents
  - evaluation
  - workflows
---

# Agent Evaluation

Agent evaluation measures whether an agent workflow reliably produces correct artifacts, not whether the model sounds plausible. The unit of evaluation is the completed task plus evidence.

A simple task success estimate is:

$$
\hat{p}_{\mathrm{success}}
=
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[\operatorname{pass}(T_i)=1]
$$

where $T_i$ is a task and $\operatorname{pass}$ is an externally checked success condition.

The estimate should be reported with uncertainty. A rough standard error for a Bernoulli pass rate is:

$$
\operatorname{SE}(\hat{p})
=
\sqrt{
\frac{\hat{p}(1-\hat{p})}{N}
}
$$

This matters because small agent eval sets can make fragile workflows look stable.

## What to Measure

- Task success under a clear rubric.
- Correctness of tool use and file edits.
- Verification coverage.
- Security and privacy violations.
- Cost, latency, and number of retries.
- Human review burden.

## Evaluation Unit

An agent task should include:

- Initial state.
- User request.
- Available tools and permissions.
- Expected artifact.
- Acceptance criteria.
- Verification evidence.
- Failure labels if it does not pass.

For coding or wiki workflows, a pass should usually require both artifact quality and a verification trail:

$$
\operatorname{pass}(T)
=
\operatorname{artifact\_ok}(T)
\land
\operatorname{evidence\_ok}(T)
\land
\operatorname{safety\_ok}(T)
$$

## Failure Taxonomy

- Planning failure: wrong decomposition, wrong order, or premature finish.
- Context failure: missed relevant file, stale memory, or hallucinated constraint.
- Tool failure: wrong command, wrong path, bad side effect, or ignored error.
- Domain failure: plausible but technically wrong content.
- Verification failure: insufficient checks or overbroad claims.
- Safety failure: secret exposure, prompt-injection obedience, or unsafe publication.
- Handoff failure: another agent or human cannot reproduce what happened.

## Coverage

Agent evaluation should cover realistic variation, not only clean demos:

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

Messy tasks include dirty worktrees, ambiguous instructions, flaky tools, partial prior work, and outdated docs. Adversarial tasks include prompt injection, malicious files, and misleading tool output.

## Checks

- Is success judged by tests, review, build output, or external ground truth?
- Are failures classified by planning, tool use, context, verification, or domain knowledge?
- Does the benchmark include realistic messy states?
- Are private data and credentials excluded from evaluation traces?
- Does the evaluation separate model quality from tool permission, scaffold quality, and verifier quality?
- Are task outcomes reproducible from the saved prompt, inputs, state, and artifacts?
- Are regressions tracked across model, prompt, tool, and policy changes?

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
