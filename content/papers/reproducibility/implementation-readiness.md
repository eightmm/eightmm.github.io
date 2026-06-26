---
title: Implementation Readiness
aliases:
  - papers/implementation-readiness
tags:
  - papers
  - reproducibility
  - implementation
---

# Implementation Readiness

Implementation readiness asks whether a paper has enough public detail to justify spending time on a reimplementation. It sits between [[papers/reproducibility/artifact-availability|Artifact availability]] and [[papers/reproducibility/reproduction-plan|Reproduction plan]].

A paper is implementation-ready only for a scoped claim:

$$
\operatorname{ready}(p,c)
=
\operatorname{artifacts}(p)
\land
\operatorname{spec}(c)
\land
\operatorname{feasible}(c)
\land
\operatorname{verifiable}(c)
$$

where $p$ is the paper and $c$ is the target claim.

## Readiness Levels

| Level | Meaning |
| --- | --- |
| `metadata-only` | The paper can be cited or tracked, but not implemented yet |
| `concept-ready` | The idea can update concept notes, but implementation details are incomplete |
| `baseline-ready` | A simplified baseline or diagnostic can be implemented |
| `claim-ready` | One named claim has enough artifacts, data, metric, and scope to test |
| `reproduction-ready` | The original result can plausibly be rerun or closely reimplemented |

## Required Fields

- Target claim: one claim, not the whole paper.
- Input/output contract: data, preprocessing, representation, target, and validity rule.
- Method specification: architecture, objective, optimizer, schedule, and inference procedure.
- Evaluation contract: benchmark, split, metric, aggregation, baseline, and failure criteria.
- Artifact status: code, data, split, config, weights, logs, predictions, and environment.
- Compute estimate: public hardware class, memory, runtime, and fallback plan.
- Risk: missing details, dependency drift, ambiguous preprocessing, or unavailable data.

## Decision Rule

Do not start a full reproduction when a smaller check can falsify the useful part:

$$
\text{start}
\iff
\text{claim-ready}
\land
\text{compute-feasible}
\land
\text{failure-informative}
$$

A failed run is useful when it can distinguish paper ambiguity, implementation error, data mismatch, compute limitation, or a false claim.

## Checks

- Is the implementation target one explicit claim?
- Are all missing artifacts marked `to verify`, `not found`, or `not applicable`?
- Can a minimum viable experiment test the claim before a full reproduction?
- Is the benchmark public and compatible with the claimed split?
- Are expected metrics and tolerances defined before running?
- Would a failure produce a useful [[papers/reproducibility/reproduction-result|Reproduction result]] note?

## Related

- [[papers/reproducibility/artifact-availability|Artifact availability]]
- [[papers/reproducibility/checklist|Reproducibility checklist]]
- [[papers/reproducibility/reproduction-plan|Reproduction plan]]
- [[papers/reproducibility/reproduction-result|Reproduction result]]
- [[concepts/research-methodology/minimum-viable-experiment|Minimum viable experiment]]
- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/run-artifact|Run artifact]]
