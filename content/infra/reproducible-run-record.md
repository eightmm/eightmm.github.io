---
title: Reproducible Run Record
tags:
  - infra
  - reproducibility
  - research-engineering
---

# Reproducible Run Record

A reproducible run record captures enough public metadata to understand what was attempted and how it was verified, without exposing private systems or unpublished results.

## Minimal Fields

- Purpose: what question the run addresses.
- Code: commit or release identifier.
- Data: public dataset version or private placeholder description.
- Split: split policy and hash if publishable.
- Config: model, objective, seed, and resource class.
- Environment: high-level software stack.
- Output: artifact type, not private path.
- Outcome: completed, failed, interrupted, or superseded.

## Reproducibility Equation

A run is not just code:

$$
\operatorname{Run}
=
(\operatorname{code},
\operatorname{data},
\operatorname{split},
\operatorname{config},
\operatorname{environment},
\operatorname{seed})
$$

If any component changes, the run identity changes.

## Checks

- Can another reader understand the setup without internal paths?
- Are private datasets, private queues, and unpublished metrics excluded?
- Is failure recorded as a useful diagnosis rather than hidden?
- Is the split policy clear enough to evaluate leakage risk?

## Related

- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[agents/verification-loop|Verification loop]]
