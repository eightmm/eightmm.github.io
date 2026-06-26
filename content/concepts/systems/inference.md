---
title: Inference
tags:
  - systems
  - inference
---

# Inference

Inference is the process of using a trained model to produce outputs. It can be offline batch inference, interactive serving, retrieval scoring, generation, or agent action selection.

The generic form is:

$$
\hat{y} = f_{\hat{\theta}}(x)
$$

For autoregressive generation:

$$
p(y_{1:T}\mid x)=\prod_{t=1}^{T}p(y_t\mid y_{<t},x)
$$

Inference cost depends on input length, output length, batch size, model size, precision, cache behavior, and hardware.

## Modes

- Batch inference for large offline datasets.
- Online inference for request-response services.
- Streaming inference for token, audio, or video outputs.
- Retrieval inference for embedding and nearest-neighbor search.
- Tool-using inference for agents.

## Inference Pipeline

Inference is usually a pipeline, not only a model call:

$$
x_{\mathrm{raw}}
\rightarrow
\operatorname{preprocess}
\rightarrow
x_{\mathrm{model}}
\rightarrow
f_{\hat{\theta}}
\rightarrow
\hat{y}_{\mathrm{raw}}
\rightarrow
\operatorname{postprocess}
\rightarrow
\hat{y}
$$

Each stage has its own validity rules and failure modes. Evaluation should use the same preprocessing and postprocessing as deployment unless the difference is explicitly tested.

## Decoding and Constraints

For generative models, inference includes decoding:

$$
\hat{y}
=
\operatorname{decode}
(p_{\hat{\theta}}(\cdot\mid x),\ c)
$$

where $c$ contains constraints such as temperature, max tokens, schemas, stop conditions, beam size, or validity functions. These settings are part of the method.

## Failure Modes

- Input is accepted but interpreted under the wrong schema.
- Preprocessing differs from training or evaluation.
- Postprocessing repairs invalid outputs without counting failures.
- Batch order, random seed, cache state, or timeout changes outputs.
- Model returns fluent but unsupported text for missing evidence.

## Checks

- Is there an [[concepts/systems/inference-contract|inference contract]] for accepted inputs, outputs, and errors?
- Is the workload latency-bound or throughput-bound?
- Is preprocessing identical to evaluation?
- Are outputs constrained to valid task objects?
- Does batching change output order, randomness, or timeout behavior?
- Are failures explicit rather than silently returning malformed outputs?
- Are decoding parameters and validity checks recorded?
- Are invalid outputs included in quality and efficiency metrics?
- Is fallback behavior explicit when retrieval, tools, or preprocessing fail?

## Related

- [[concepts/systems/inference-contract|Inference contract]]
- [[concepts/systems/model-card|Model card]]
- [[concepts/tasks/index|Tasks]]
- [[concepts/tasks/task-output-space|Task output space]]
- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/llm/decoding|Decoding]]
- [[concepts/llm/structured-output|Structured output]]
- [[concepts/systems/batch-online-inference|Batch and online inference]]
- [[concepts/systems/model-serving|Model serving]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[infra/inference/serving|Inference serving]]
