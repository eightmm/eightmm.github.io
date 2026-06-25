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

## Checks

- Can a later note trace a claim back to a run?
- Are private paths, credentials, hostnames, and unpublished results excluded from public notes?
- Are metric definitions stable across runs?
- Are failed and negative results recorded when they change the next decision?
- Is the run record light enough to use consistently?

## Related

- [[concepts/systems/experiment-lifecycle|Experiment lifecycle]]
- [[concepts/systems/run-artifact|Run artifact]]
- [[concepts/systems/training-run|Training run]]
- [[concepts/machine-learning/learning-curve|Learning curve]]
- [[concepts/machine-learning/validation-curve|Validation curve]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[infra/reproducible-run-record|Reproducible run record]]
- [[logs/public-log-format|Public log format]]
- [[projects/project-milestone-format|Project milestone format]]
