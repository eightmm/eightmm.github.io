---
title: Completion Audit
tags:
  - agents
  - verification
  - workflows
---

# Completion Audit

A completion audit checks whether an agent is allowed to say "done." It is stricter than running tests: it compares the full objective against current evidence.

Given requirements $\mathcal{R}$ and evidence ledger $L$, completion requires:

$$
\operatorname{complete}
=
\bigwedge_{r \in \mathcal{R}}
\operatorname{covered}(r, L)
$$

If any requirement is unverified, contradicted, too broad for the evidence, or missing, the task is not complete.

## Audit Steps

1. Restate the actual objective without narrowing it to the completed work.
2. List explicit deliverables, commands, invariants, safety rules, and reporting requirements.
3. For each requirement, name the evidence that would prove it.
4. Inspect current files, command output, rendered artifacts, or external state.
5. Mark each requirement as `proved`, `contradicted`, `incomplete`, `weak evidence`, or `missing`.
6. Continue work unless every required item is proved.

## Common Failure Modes

- Treating a build as proof that content is correct.
- Treating old memory as current evidence.
- Reporting a skipped check as if it passed.
- Completing a smaller task than the user requested.
- Ignoring public-safety or privacy requirements.
- Forgetting to push after a repository change when the workflow requires it.

## Checks

- Did the audit preserve the original scope?
- Does every requirement have direct evidence?
- Are broad claims supported by broad checks?
- Are uncertain items explicitly marked as not verified?
- Is the final answer consistent with the newest user request?

## Related

- [[agents/verification/evidence-ledger|Evidence ledger]]
- [[agents/verification/acceptance-criteria|Acceptance criteria]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/core/agent-operating-contract|Agent operating contract]]
- [[agents/workflows/agent-runbook|Agent runbook]]
