---
title: Verification Loop
tags:
  - agents
  - llm
  - verification
---

# Verification Loop

A verification loop closes the gap between "the agent said it's done" and "the artifact is correct." After each change, the agent runs a concrete check and feeds the result back into the next decision.

The loop compares a claim $c$ against evidence $E$:

$$
\operatorname{verified}(c)
=
\exists E\; \text{such that}\; E \Rightarrow c
$$

If the evidence is missing, too narrow, stale, or indirect, the claim is not verified.

## Verification Ladder

- Syntax or format check.
- Narrow unit or link check.
- Build or integration check.
- Runtime smoke test or rendered-page check.
- Review for security, privacy, data leakage, or scientific validity.

## Practical Checks

- End every change with a real check: build, test, lint, or smoke run.
- Use the narrowest useful check first, then broaden if risk warrants.
- Treat a failing check as the signal to fix, not to retry the same action.
- Never report success on a skipped or impossible check — say "not verified."
- Capture the command and output so the result is reproducible.
- Match the check to the claim. A build passing does not prove content accuracy.
- Treat a green check as evidence only for the behavior it actually covers.

## Related

- [[agents/core/agent-operating-contract|Agent operating contract]]
- [[agents/core/agent-loop|Agent loop]]
- [[agents/verification/acceptance-criteria|Acceptance criteria]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[agents/core/planning|Planning]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[agents/tools/tool-use|Tool use]]
- [[concepts/evaluation/index|Evaluation]]
