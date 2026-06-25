---
title: Evidence Ledger
tags:
  - agents
  - verification
  - evidence
---

# Evidence Ledger

An evidence ledger records what an agent checked before making a claim. It prevents a final answer from relying on memory, intent, or a passing check that does not cover the actual requirement.

For a claim $c$, the ledger should store:

$$
L(c)
=
\{(r_i, e_i, v_i, t_i)\}_{i=1}^{n}
$$

where $r_i$ is a requirement, $e_i$ is evidence, $v_i$ is the verification method, and $t_i$ is the time or state of the check.

## Evidence Types

| Evidence | Proves |
| --- | --- |
| File diff | What changed |
| Command output | Build, test, lint, or scan result |
| Rendered page check | Whether generated output links or displays correctly |
| Source citation | Whether a fact is grounded |
| Review note | Human or second-agent judgment |
| Run artifact | Config, log, metric, prediction, or checkpoint |

## Ledger Fields

- Requirement: the explicit condition being checked.
- Evidence source: file, command, rendered output, source document, or reviewer.
- Coverage: what the evidence proves and what it does not prove.
- Result: pass, fail, inconclusive, skipped, or blocked.
- Freshness: whether the evidence comes from the current state.
- Public boundary: whether evidence can be quoted or must be summarized.

## Checks

- Does every completion claim point to evidence?
- Is the evidence current, direct, and broad enough?
- Are skipped or impossible checks marked `not verified`?
- Does a passing build avoid being overused as proof of content accuracy?
- Are private logs, paths, hostnames, account names, and unpublished results omitted?

## Related

- [[agents/verification/acceptance-criteria|Acceptance criteria]]
- [[agents/verification/verification-loop|Verification loop]]
- [[agents/verification/completion-audit|Completion audit]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[logs/sanitization-checklist|Sanitization checklist]]
