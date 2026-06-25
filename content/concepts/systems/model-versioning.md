---
title: Model Versioning
tags:
  - systems
  - serving
  - reproducibility
---

# Model Versioning

Model versioning records which model artifact, code, configuration, data boundary, and runtime produced a result. It is the serving-time counterpart of [[concepts/systems/run-artifact|run artifact]] and the public counterpart of a [[concepts/systems/model-card|model card]].

A useful model version is a tuple:

$$
v
=
(\theta, c, d, \phi, \psi, e)
$$

where $\theta$ is the checkpoint or weights, $c$ is code/config, $d$ is the data or benchmark boundary, $\phi$ is preprocessing, $\psi$ is postprocessing, and $e$ is the environment.

## What Must Travel Together

- Checkpoint or exported model artifact.
- Architecture and runtime configuration.
- Tokenizer, featurizer, graph builder, structure preparation, or normalization state.
- Postprocessing and validity checks.
- Intended task and [[concepts/systems/inference-contract|inference contract]].
- Evaluation evidence and limitations.
- Environment constraints: precision, library versions, hardware class, memory envelope, and context length when relevant.

## Version Compatibility

Two versions may share weights but differ in preprocessing or runtime. They should still be treated as different deployed versions:

$$
f_v(x)
=
\psi_v(f_{\theta_v}(\phi_v(x)))
$$

Changing $\phi_v$ or $\psi_v$ can change outputs even when $\theta_v$ is unchanged.

## Checks

- Can an output be traced back to one exact version?
- Are preprocessing and postprocessing versioned with the checkpoint?
- Is the model version distinct from a git commit when external artifacts are involved?
- Is the version linked to evaluation evidence without exposing private paths or unpublished metrics?
- Can [[concepts/systems/deployment-strategy|deployment strategy]] roll back to the previous version?

## Related

- [[concepts/systems/run-artifact|Run artifact]]
- [[concepts/systems/model-card|Model card]]
- [[concepts/systems/inference-contract|Inference contract]]
- [[concepts/systems/deployment-strategy|Deployment strategy]]
- [[concepts/systems/reproducibility|Reproducibility]]
- [[projects/project-artifact-release|Project artifact release]]
