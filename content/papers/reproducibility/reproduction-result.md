---
title: Reproduction Result
unlisted: true
aliases:
  - papers/reproduction-result
tags:
  - papers
  - reproducibility
  - experiments
---

# Reproduction Result

A reproduction result records what happened when a paper claim was rerun, reimplemented, or tested with a smaller public experiment. It should be narrower than a project log and more structured than a casual note.

The result should connect:

$$
(\text{claim}, \text{plan}, \text{run}, \text{artifact}, \text{metric}, \text{decision})
$$

so the paper note can say what evidence was checked and what remains unresolved.

## Result Types

- Rerun: using public code and artifacts from the paper.
- Reimplementation: rebuilding the method from the description.
- Diagnostic reproduction: testing one component, ablation, preprocessing step, or metric.
- Negative result: a failed or contradictory result with a useful explanation.
- Inconclusive result: the run failed to answer the target claim.

## Template

| Field | Content |
| --- | --- |
| Paper | Wikilink to the paper note |
| Target claim | The exact scoped claim tested |
| Plan | Link to [[papers/reproducibility/reproduction-plan|Reproduction plan]] |
| Run record | Public-safe run id, config summary, seed policy, and environment class |
| Artifact | Logs, metrics, predictions, checkpoint, or `not released` |
| Metric | Metric definition, split, aggregation, and tolerance |
| Outcome | `supports`, `contradicts`, `inconclusive`, or `blocked by missing artifact` |
| Limitation | What the run does not prove |
| Next decision | Stop, retry, simplify, compare baseline, or promote to project |

## Interpretation

The outcome should not be stronger than the evidence:

$$
\text{claim strength}
\le
f(\text{artifact quality}, \text{split match}, \text{metric match}, \text{compute match})
$$

A reproduction on a smaller dataset can check implementation logic, but it should not be written as proof of the original benchmark result.

## Checks

- Was the target claim defined before running?
- Does the run use public data or a clearly marked public-safe substitute?
- Are failures classified rather than hidden?
- Are private paths, hostnames, account names, and unpublished metrics removed?
- Can the metric be recomputed from public artifacts or described outputs?
- Is the next decision explicit?

## Related

- [[papers/reproducibility/reproduction-plan|Reproduction plan]]
- [[papers/reproducibility/implementation-readiness|Implementation readiness]]
- [[papers/reproducibility/checklist|Reproducibility checklist]]
- [[concepts/research-methodology/negative-result|Negative result]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[infra/reproducibility/run-record|Reproducible run record]]
