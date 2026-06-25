---
title: Reflection and Critique
tags:
  - agents
  - verification
  - workflows
---

# Reflection and Critique

Reflection and critique ask an agent to inspect a plan, output, or failure before continuing. It can improve quality, but only when tied to evidence and concrete checks.

A critique step can be modeled as:

$$
c_t = \operatorname{Critique}(a_t, E_t, R)
$$

where $a_t$ is the proposed action or artifact, $E_t$ is evidence, and $R$ is the review rubric.

## Useful Critique Targets

- Missing requirements.
- Broken links, tests, or build steps.
- Unsupported claims.
- Risky side effects.
- Inconsistent terminology.
- Public/private information leakage.

## Failure Modes

- The critique becomes generic advice.
- The agent rationalizes its own output.
- The critique checks style while missing factual errors.
- The loop keeps revising without new evidence.

## Checks

- Does critique cite the artifact being reviewed?
- Does it produce actionable changes?
- Is there an external verification step after critique?
- Would an independent reviewer catch different errors?
- Does the workflow stop when no new evidence is being added?

## Related

- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[concepts/evaluation/error-analysis|Error analysis]]
- [[papers/paper-review-workflow|Paper review workflow]]
