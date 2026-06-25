---
title: Run Artifact
tags:
  - systems
  - artifacts
  - reproducibility
---

# Run Artifact

A run artifact is any durable output that lets a later reader inspect, resume, compare, or verify a run. It is the systems-side counterpart of [[papers/artifact-availability|Artifact availability]].

For a run $r$, an artifact bundle can be represented as:

$$
B(r)
=
\{\text{config}, \text{logs}, \text{metrics}, \text{predictions}, \text{checkpoint}, \text{environment}, \text{notes}\}
$$

where each element should be versioned or marked `to verify`.

## Artifact Types

- Config: model, data, optimizer, schedule, precision, seed policy, and runtime options.
- Logs: training loss, validation metrics, warnings, failure reason, and restart history.
- Metrics: metric definition, aggregation rule, split name, and selection criterion.
- Predictions: per-example outputs when public and safe to share.
- Checkpoint: model weights plus optimizer and scheduler state when resume matters.
- Environment: package versions, accelerator/runtime assumptions, and container or lockfile.
- Notes: hypothesis, expected outcome, observed result, interpretation, and next decision.

## Minimal Public Bundle

Public blog notes should usually publish the shape of the artifact bundle, not private locations or raw internal records:

| Artifact | Public note should include |
| --- | --- |
| Config | Generic settings or fields, not private paths |
| Logs | Metric definitions and summary, not private console dumps |
| Metrics | Split, metric, aggregation rule, confidence if available |
| Predictions | Only if data and outputs are public |
| Checkpoint | Public release link or `not released` |
| Environment | Package/runtime class, not hostnames |
| Notes | Decision and limitation, not internal task names |

## Checks

- Can a later reader tell what was run?
- Can the metric be recomputed from public predictions or described outputs?
- Can the run resume from checkpoint state if interruption matters?
- Are failed runs recorded without exposing private infrastructure?
- Are artifacts tied to a code version, config, data version, and seed policy?
- Is every missing artifact marked `to verify`, `not released`, or `not applicable`?

## Related

- [[concepts/systems/experiment-lifecycle|Experiment lifecycle]]
- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/checkpoint-state|Checkpoint state]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[papers/artifact-availability|Artifact availability]]
- [[infra/reproducible-run-record|Reproducible run record]]
- [[logs/public-log-format|Public log format]]
