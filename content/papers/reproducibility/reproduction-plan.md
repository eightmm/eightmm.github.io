---
title: Reproduction Plan
unlisted: true
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

## Run Ladder

Prefer the cheapest informative step:

| Step | Purpose | Stop If |
|---|---|---|
| artifact audit | check what exists | critical artifact missing |
| metric-only check | rerun metric on released predictions/samples | reported number cannot be matched |
| preprocessing check | reproduce processed data or split | counts or labels diverge |
| baseline rerun | verify benchmark and compute path | baseline cannot be reproduced |
| small-scale implementation | test method mechanics | loss/metric behavior contradicts claim |
| full-scale rerun | test original scale | earlier checks support the cost |

This avoids spending compute before the benchmark contract is understood.

## Failure Taxonomy

| Failure | Meaning |
|---|---|
| artifact missing | claim cannot be checked from public materials |
| data mismatch | dataset, filtering, split, or preprocessing differs |
| implementation ambiguity | method details are insufficient |
| compute infeasible | required resources exceed the planned budget |
| metric mismatch | evaluation script or aggregation differs |
| claim not supported | run contradicts the stated paper interpretation |

## Checks

- Is the reproduction testing one claim rather than the whole paper?
- Are missing paper details marked explicitly?
- Can the run be done on public data?
- Is the expected compute realistic?
- What result would falsify the paper note's interpretation?
- What result should be recorded as `inconclusive` rather than success or failure?
- Is the stop rule defined before running?
- Are failed, invalid, or filtered outputs included in the result record?

## Related

- [[papers/reproducibility/checklist|Reproducibility checklist]]
- [[papers/reproducibility/implementation-readiness|Implementation readiness]]
- [[papers/reproducibility/reproduction-result|Reproduction result]]
- [[papers/reproducibility/artifact-availability|Artifact availability]]
- [[papers/analysis/benchmark-card|Benchmark card]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[concepts/research-methodology/minimum-viable-experiment|Minimum viable experiment]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[infra/reproducibility/run-record|Reproducible run record]]
