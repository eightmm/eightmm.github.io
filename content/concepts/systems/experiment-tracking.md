---
title: Experiment Tracking
tags:
  - systems
  - experiments
  - reproducibility
---

# Experiment Tracking

Experiment tracking records what was run, under which configuration, and what happened. It prevents research logs from becoming disconnected screenshots or unverifiable claims.

A minimal run record is:

$$
r = (\text{commit}, \text{config}, \text{data version}, \text{seed}, \text{environment}, \text{metrics}, \text{artifacts})
$$

## What To Track

- Run identifier.
- Experiment lifecycle stage and decision question.
- Code commit and diff state.
- Dataset and benchmark version.
- Configuration and hyperparameters.
- Hardware and environment summary.
- Metrics, artifacts, and checkpoints.
- Learning curves, validation curves, and selection decisions.
- Failure reason when the run fails.

## Decision Link

Tracking is useful only if it connects a run to a decision:

$$
\text{question}
\rightarrow
\text{hypothesis}
\rightarrow
\text{run}
\rightarrow
\text{artifact}
\rightarrow
\text{decision}
$$

A metric without the question and selection rule is just a number. A failed run without a failure reason cannot teach the next run anything.

## Run Table

A compact tracking table should include:

$$
r_i
=
(\text{id},
\text{status},
\text{claim},
\text{config hash},
\text{data hash},
\text{metric summary},
\text{artifact pointer},
\text{next decision})
$$

For public notes, artifact pointers should be public-safe summaries or release artifacts, not private paths.

## Failure Modes

- Dashboards contain metrics but not config or data version.
- Screenshots are used as evidence without a run artifact.
- Failed runs disappear, biasing the narrative.
- Multiple runs compete for the same claim without a final selection rule.
- Public writeups expose private paths, hostnames, usernames, or unpublished results.

## Checks

- Can a later note trace a claim back to a run?
- Are private paths, credentials, hostnames, and unpublished results excluded from public notes?
- Are metric definitions stable across runs?
- Are failed and negative results recorded when they change the next decision?
- Is the run record light enough to use consistently?
- Does each tracked run state the question it answers?
- Is the final selected checkpoint or model tied to a validation rule?
- Are artifacts sufficient to reproduce, inspect, or reject the claim?

## Related

- [[concepts/systems/experiment-lifecycle|Experiment lifecycle]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[concepts/systems/model-versioning|Model versioning]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/training-run|Training run]]
- [[concepts/machine-learning/learning-curve|Learning curve]]
- [[concepts/machine-learning/validation-curve|Validation curve]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[infra/reproducibility/run-record|Reproducible run record]]
- [[logs/public-log-format|Public log format]]
- [[projects/project-milestone-format|Project milestone format]]
