---
title: Reproduction Plan
aliases:
  - papers/reproduction-plan
tags:
  - papers
  - reproducibility
  - workflows
---

# Reproduction Plan

A reproduction plan turns a paper into the smallest public-safe implementation or rerun that could check the main claim.

Create this only after checking [[papers/reproducibility/implementation-readiness|Implementation readiness]]. If the run happens, record the outcome as [[papers/reproducibility/reproduction-result|Reproduction result]].

The plan should minimize scope:

$$
R^\star
= \arg\min_{R \in \mathcal{R}}
\operatorname{cost}(R)
+ \lambda \operatorname{risk}(R)
\quad
\text{s.t. } R \text{ tests claim } c
$$

$R$ is a candidate reproduction, $c$ is the target claim, and $\lambda$ controls how much implementation risk matters.

## Plan Sections

- Target claim: the one claim to reproduce.
- Minimal dataset or benchmark: public and small enough to run.
- Baseline: simplest comparison required.
- Required artifacts: code, config, data split, model weights, or preprocessing.
- Environment: package versions, hardware class, and expected runtime.
- Verification: metric, tolerance, and failure criteria.
- Stop rule: when to stop reproducing and record the limitation.

## Checks

- Is the reproduction testing one claim rather than the whole paper?
- Are missing paper details marked explicitly?
- Can the run be done on public data?
- Is the expected compute realistic?
- What result would falsify the paper note's interpretation?
- What result should be recorded as `inconclusive` rather than success or failure?

## Related

- [[papers/reproducibility/checklist|Reproducibility checklist]]
- [[papers/reproducibility/implementation-readiness|Implementation readiness]]
- [[papers/reproducibility/reproduction-result|Reproduction result]]
- [[papers/reproducibility/artifact-availability|Artifact availability]]
- [[papers/analysis/benchmark-card|Benchmark card]]
- [[concepts/research-methodology/minimum-viable-experiment|Minimum viable experiment]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[infra/reproducibility/run-record|Reproducible run record]]
