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

## Checks

- Is there an [[concepts/systems/inference-contract|inference contract]] for accepted inputs, outputs, and errors?
- Is the workload latency-bound or throughput-bound?
- Is preprocessing identical to evaluation?
- Are outputs constrained to valid task objects?
- Does batching change output order, randomness, or timeout behavior?
- Are failures explicit rather than silently returning malformed outputs?

## Related

- [[concepts/systems/inference-contract|Inference contract]]
- [[concepts/systems/model-card|Model card]]
- [[concepts/tasks/index|Tasks]]
- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/systems/batch-online-inference|Batch and online inference]]
- [[concepts/systems/model-serving|Model serving]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[infra/inference-serving|Inference serving]]
