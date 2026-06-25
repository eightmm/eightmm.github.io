---
title: Agent Verification
tags:
  - agents
  - verification
---

# Agent Verification

Verification notes describe how agent outputs are checked, reviewed, constrained, and kept safe for public workflows.

Verification turns an agent's claim into evidence. The core relation is:

$$
\operatorname{verified}(c)
=
\exists E\ \text{such that}\ E \Rightarrow c
$$

where $c$ is the claim and $E$ is evidence from tests, builds, rendered pages, logs, source inspection, review, or human judgment. A broad claim needs broad evidence; a narrow check only proves the narrow behavior it covers.

## Verification Ladder

1. Define [[agents/verification/acceptance-criteria|acceptance criteria]] before judging the output.
2. Collect evidence in an [[agents/verification/evidence-ledger|evidence ledger]].
3. Run a [[agents/verification/verification-loop|verification loop]] after each meaningful side effect.
4. Use [[agents/verification/reflection-and-critique|reflection and critique]] to look for missing checks.
5. Use [[agents/verification/completion-audit|completion audit]] before claiming a broad task is done.

## Notes

- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/acceptance-criteria|Acceptance criteria]]
- [[agents/verification/evidence-ledger|Evidence ledger]]
- [[agents/verification/completion-audit|Completion audit]]
- [[agents/verification/reflection-and-critique|Reflection and critique]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[agents/verification/human-in-the-loop|Human in the loop]]
- [[agents/verification/prompt-injection|Prompt injection]]

## Checks

- What exact claim is being verified?
- Is the evidence direct, current, and scoped to that claim?
- What did each check prove, and what did it not prove?
- Was any check skipped, impossible, or too narrow?
- Does the output expose private infrastructure, collaborators, credentials, or unpublished results?
- Does the final summary separate verified facts from unverified assumptions?

## Related

- [[agents/index|Agents]]
- [[agents/tools/tool-contract|Tool contract]]
- [[agents/tools/tool-result-handling|Tool result handling]]
- [[agents/core/agent-operating-contract|Agent operating contract]]
- [[concepts/llm/hallucination-grounding|Hallucination and grounding]]
- [[logs/sanitization-checklist|Sanitization checklist]]
