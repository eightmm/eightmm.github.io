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

## What to Measure

- Task success under a clear rubric.
- Correctness of tool use and file edits.
- Verification coverage.
- Security and privacy violations.
- Cost, latency, and number of retries.
- Human review burden.

## Checks

- Is success judged by tests, review, build output, or external ground truth?
- Are failures classified by planning, tool use, context, verification, or domain knowledge?
- Does the benchmark include realistic messy states?
- Are private data and credentials excluded from evaluation traces?

## Related

- [[agents/verification-loop|Verification loop]]
- [[agents/multi-agent-review|Multi-agent review]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/learning/reward-modeling|Reward modeling]]
- [[concepts/learning/preference-optimization|Preference optimization]]
- [[agents/human-in-the-loop|Human in the loop]]
