---
title: Reproducible Run Record
aliases:
  - infra/reproducible-run-record
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
- Claim support: public claim or `not used for a claim`.
- Reconciliation: final scheduler/artifact/log state when the run used shared compute.

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

## Public vs Private Fields

Run record는 공개 가능한 정보와 내부 운영 정보를 분리해야 합니다.

| Public field | Private-only field |
| --- | --- |
| resource class | host name, node name, account name |
| artifact type | absolute private path |
| software stack | internal module path |
| public dataset/version | private dataset location |
| failure class | raw log with credentials or user names |
| commit/hash | SSH endpoint, port, credential |

Public note는 재현 가능한 reasoning을 남기되, infrastructure attack surface나 unpublished result를 노출하지 않습니다.

## Closeout States

| State | Meaning |
| --- | --- |
| completed | artifacts checked and recorded |
| failed | failure class and next action recorded |
| interrupted | checkpoint or partial output exists |
| superseded | replaced by a later run |
| not-for-claim | useful operational record but not evidence for a public claim |

Do not treat an unverified completed process as a completed run.

## Evidence Links

For public wiki, evidence can be generic:

| Evidence | Public-safe form |
| --- | --- |
| build result | command name and pass/fail summary |
| scheduler state | terminal category, not private job identifier |
| artifact check | artifact type and validation rule |
| metric | only if already public and non-sensitive |
| failure | generalized failure mode and fix |

## Checks

- Can another reader understand the setup without internal paths?
- Are private datasets, private queues, and unpublished metrics excluded?
- Is failure recorded as a useful diagnosis rather than hidden?
- Is the split policy clear enough to evaluate leakage risk?
- Is the final outcome supported by logs, artifacts, and scheduler state?
- Is the record explicit about whether it supports a public claim?
- Are private operational fields excluded or generalized?

## Related

- [[infra/hpc/job-lifecycle|HPC job lifecycle]]
- [[infra/hpc/job-reconciliation|Job reconciliation]]
- [[infra/hpc/checkpointing|Checkpointing]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/research-methodology/claim-evidence-record|Claim evidence record]]
- [[agents/verification/verification-loop|Verification loop]]
