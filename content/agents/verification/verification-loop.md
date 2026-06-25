---
title: Verification Loop
tags:
  - agents
  - llm
  - verification
---

# Verification Loop

A verification loop closes the gap between "the agent said it's done" and "the artifact is correct." After each change, the agent runs a concrete check and feeds the result back into the next decision.

## Practical Checks

- End every change with a real check: build, test, lint, or smoke run.
- Use the narrowest useful check first, then broaden if risk warrants.
- Treat a failing check as the signal to fix, not to retry the same action.
- Never report success on a skipped or impossible check — say "not verified."
- Capture the command and output so the result is reproducible.

## Related

- [[agents/core/agent-loop|Agent loop]]
- [[agents/verification/agent-evaluation|Agent evaluation]]
- [[agents/core/planning|Planning]]
- [[agents/workflows/multi-agent-review|Multi-agent review]]
- [[agents/tools/tool-use|Tool use]]
- [[concepts/evaluation/index|Evaluation]]
