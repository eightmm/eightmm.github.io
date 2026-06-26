---
title: Paper Reproducibility
tags:
  - papers
  - reproducibility
---

# Paper Reproducibility

Paper reproducibility notes decide whether a paper has enough public material to rerun, reimplement, or compare a scoped claim.

Reproduction should be scoped to a claim, not to an entire paper:

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

where $p$ is the paper and $c$ is the claim to check.

## Scope

- Public artifact availability.
- Reproducibility checklists and implementation readiness.
- Minimum reproduction plans and reproduction-result records.
- Public-safe evidence for reruns, reimplementations, and diagnostic checks.

## Notes

- [[papers/reproducibility/artifact-availability|Artifact availability]]
- [[papers/reproducibility/checklist|Reproducibility checklist]]
- [[papers/reproducibility/implementation-readiness|Implementation readiness]]
- [[papers/reproducibility/reproduction-plan|Reproduction plan]]
- [[papers/reproducibility/reproduction-result|Reproduction result]]

## Checks

- Are code, data, splits, config, weights, logs, predictions, and environment checked separately?
- Is the target claim narrow enough to test with public artifacts?
- Is the minimum viable experiment defined before spending compute?
- Does the result state success, contradiction, inconclusive outcome, or diagnostic-only value?
- Are private datasets, private paths, unpublished metrics, and collaborator details excluded?

## Where New Notes Go

- Paper-specific artifact and reproduction notes go here.
- General experiment design goes under [[concepts/research-methodology/minimum-viable-experiment|Minimum viable experiment]].
- Run artifact structure goes under [[concepts/systems/run-artifact|Run artifact]].
- Public operational run records go under [[infra/reproducibility/index|Reproducibility infra]].

## Related

- [[concepts/systems/reproducibility|Reproducibility]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[concepts/research-methodology/minimum-viable-experiment|Minimum viable experiment]]
- [[papers/analysis/index|Paper analysis]]
