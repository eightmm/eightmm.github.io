---
title: Deployment Strategy
tags:
  - systems
  - serving
  - deployment
---

# Deployment Strategy

A deployment strategy defines how a trained model version becomes available to users, jobs, or downstream workflows. It connects [[concepts/systems/model-serving|model serving]], [[concepts/systems/inference-contract|inference contract]], [[concepts/systems/model-versioning|model versioning]], and [[concepts/systems/observability|observability]].

Deployment is not only copying weights into a runtime. It is a controlled change to a system:

$$
\Delta S
=
(\Delta \theta, \Delta \phi, \Delta \psi, \Delta E, \Delta R)
$$

where $\Delta \theta$ is a model-parameter change, $\Delta \phi$ is preprocessing, $\Delta \psi$ is postprocessing, $\Delta E$ is the runtime environment, and $\Delta R$ is routing or request policy.

## Common Patterns

- Direct replacement: simplest, but highest risk if rollback is weak.
- Blue-green deployment: keep old and new stacks separate, then switch traffic.
- Canary deployment: send a small traffic fraction to the new version first.
- Shadow deployment: run the new version on copied requests without affecting responses.
- Offline batch rollout: score a fixed dataset or queue before online exposure.

For a canary rollout, traffic routing can be written as:

$$
y(x)
=
\begin{cases}
f_{\mathrm{new}}(x) & \text{with probability } \alpha \\
f_{\mathrm{old}}(x) & \text{with probability } 1-\alpha
\end{cases}
$$

where $\alpha$ is the rollout fraction.

## Checks

- Is the new version tied to its preprocessing, postprocessing, config, and environment?
- Is the acceptance criterion defined before rollout?
- Are quality, latency, error rate, and invalid-output rate monitored separately?
- Is rollback a documented action rather than an improvised fix?
- Does the rollout avoid logging private requests or unpublished results?
- Does the deployment preserve the same assumptions used in [[concepts/evaluation/evaluation-protocol|evaluation protocol]]?

## Related

- [[concepts/systems/model-versioning|Model versioning]]
- [[concepts/systems/model-serving|Model serving]]
- [[concepts/systems/inference-contract|Inference contract]]
- [[concepts/systems/observability|Observability]]
- [[concepts/systems/failure-recovery|Failure recovery]]
- [[infra/inference/serving|Inference serving]]
