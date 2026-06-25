---
title: Inference Contract
tags:
  - systems
  - inference
  - contracts
---

# Inference Contract

An inference contract defines what an inference workflow accepts, what it returns, which outputs are valid, and how failures are reported. It is the runtime counterpart of [[concepts/tasks/task-specification|Task specification]].

A contract can be written as:

$$
I
=
(\mathcal{X}_{\mathrm{request}}, \phi, f_{\hat{\theta}}, \psi, \mathcal{Y}_{\mathrm{response}}, E)
$$

where $\mathcal{X}_{\mathrm{request}}$ is the accepted request space, $\phi$ is preprocessing, $f_{\hat{\theta}}$ is the trained model, $\psi$ is postprocessing, $\mathcal{Y}_{\mathrm{response}}$ is the response space, and $E$ is the explicit error set.

## Contract Fields

- Input schema: required fields, optional fields, units, types, shapes, and allowed ranges.
- Preprocessing: tokenizer, featurizer, resizing, graph construction, normalization, or structure preparation.
- Model version: checkpoint, config, preprocessing version, and runtime environment.
- Output schema: class, score, ranking, sequence, structured object, coordinates, action, or evidence fields.
- Validity rule: syntax, schema, geometry, threshold, confidence, or safety boundary.
- Error format: unsupported input, invalid output, timeout, missing dependency, low confidence, or internal failure.
- Resource envelope: latency, throughput, memory, batch size, context length, or token budget.
- Logging boundary: what can be logged publicly and what must be excluded.

## Response Validity

The response should satisfy:

$$
\hat{y}
=
\psi(f_{\hat{\theta}}(\phi(x)))
\in
\mathcal{Y}_{\mathrm{valid}}(x)
\cup
E
$$

where $\mathcal{Y}_{\mathrm{valid}}(x)$ is the valid response set for request $x$. Invalid outputs should be rejected, repaired with a documented policy, or returned as explicit errors.

## Checks

- Can another workflow construct a valid request without private context?
- Are preprocessing and postprocessing tied to the model version?
- Are input validation rules stated before model execution?
- Are unsupported inputs rejected explicitly?
- Are invalid outputs counted as failures rather than silently filtered?
- Are confidence, uncertainty, and abstention policies defined when needed?
- Are latency, timeout, and batching assumptions stated?
- Does logging avoid private inputs, credentials, and unpublished data?

## Related

- [[concepts/systems/model-card|Model card]]
- [[concepts/systems/model-versioning|Model versioning]]
- [[concepts/systems/data-validation|Data validation]]
- [[concepts/systems/inference|Inference]]
- [[concepts/systems/model-serving|Model serving]]
- [[concepts/systems/batch-online-inference|Batch and online inference]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[concepts/tasks/task-specification|Task specification]]
- [[concepts/tasks/task-output-space|Task output space]]
- [[concepts/evaluation/failure-mode-taxonomy|Failure mode taxonomy]]
- [[concepts/llm/structured-output|Structured output]]
- [[agents/tools/tool-contract|Tool contract]]
