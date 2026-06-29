---
title: Inference Serving
aliases:
  - infra/inference-serving
tags:
  - systems
  - inference
  - serving
---

# Inference Serving

Inference serving turns a trained model into a low-latency, high-throughput service. For LLMs the dominant costs are KV-cache memory and the prefill/decode split; batching and quantization are the main levers.

Serving should start from a capacity plan:

$$
(\text{request shape}, \text{concurrency}, \text{latency target}, \text{memory budget})
\rightarrow
\text{batching and deployment policy}
$$

## Serving Object

A served model is not only a checkpoint. It is the checkpoint plus the input contract, preprocessing, batching policy, runtime, and failure behavior.

$$
\text{served model}
=
(\text{weights}, \text{preprocess}, \text{runtime}, \text{batching}, \text{postprocess}, \text{failure policy})
$$

If any element changes, latency, quality, and validity can change even when the model name stays the same.

## Request Shape

Capacity planning starts with the request distribution, not the average prompt.

| Field | Why |
| --- | --- |
| input length | controls prefill and memory |
| output length | controls decode time and KV cache growth |
| concurrency | controls queueing and memory pressure |
| payload size | controls network and preprocessing |
| valid input range | controls rejection and fallback behavior |
| response mode | synchronous, async, streaming, or batch |

For decoder-style generation:

$$
t_{\mathrm{request}}
\approx
t_{\mathrm{queue}}
+ t_{\mathrm{prefill}}(T_{\mathrm{in}})
+ T_{\mathrm{out}} t_{\mathrm{decode}}
+ t_{\mathrm{postprocess}}
$$

Throughput claims should state whether they count requests/sec, samples/sec, tokens/sec, or jobs/day.

## Memory Budget

For autoregressive serving, request concurrency is often constrained by weights plus request-specific cache:

$$
M_{\mathrm{total}}
\approx
M_{\mathrm{weights}}
+ M_{\mathrm{runtime}}
+ N_{\mathrm{active}} M_{\mathrm{request}}
$$

where $M_{\mathrm{request}}$ may include KV cache, activations, buffers, tokenizer state, and framework overhead.

The KV-cache term grows with layer count, heads, context length, and dtype:

$$
M_{\mathrm{KV}}
\propto
L \cdot H \cdot T \cdot d_{\mathrm{head}} \cdot b
$$

This is why context length and concurrency must be part of the serving contract.

## Batching Modes

| Mode | Use For | Risk |
| --- | --- | --- |
| no batching | lowest complexity, debugging, low traffic | poor GPU utilization |
| static batching | offline or fixed-shape workloads | idle time and padding waste |
| dynamic batching | online serving with variable arrival | tail latency and queue tuning |
| continuous batching | autoregressive serving | scheduler complexity and memory pressure |
| offline batch | large backfills or evaluation | not representative of online latency |

## Quality Boundary

Serving optimizations can change the model's behavior. Treat them as system changes, not only performance changes.

| Optimization | Check |
| --- | --- |
| quantization | calibration, task metric, failure examples |
| truncation | lost context and unsupported input handling |
| caching | stale data, privacy boundary, cache invalidation |
| batching | order dependence, timeout, tail latency |
| fallback model | output compatibility and evaluation boundary |

## Practical Checks

- Distinguish latency-bound (single request) from throughput-bound (batched) goals.
- Use continuous/dynamic batching to keep the GPU busy without hurting tail latency.
- Size the KV cache against context length and concurrency.
- Apply quantization (int8/fp8) only after measuring accuracy impact.
- Measure p50/p95/p99 latency, not just averages.
- Use [[concepts/systems/inference-capacity-planning|Inference capacity planning]] before choosing hardware or batching policy.

## Public Serving Note Checklist

| Item | Include |
| --- | --- |
| model identity | public model name or generic model class |
| input contract | schema, limits, invalid input handling |
| output contract | schema, confidence, error format |
| workload shape | context length, batch/concurrency, request rate |
| capacity target | latency, throughput, memory, cost |
| quality gate | metric or manual evaluation affected by serving changes |
| observability | public-safe aggregate metrics, not raw private input |

## Boundary

Keep this note about serving behavior and capacity. Put GPU memory diagnosis in [[infra/gpu/index|GPU]], endpoint rollout in [[concepts/systems/deployment-strategy|Deployment strategy]], and public user-facing validity in [[concepts/systems/inference-contract|Inference contract]] or [[concepts/systems/model-card|Model card]].

## Related

- [[infra/gpu/index|GPU]]
- [[infra/gpu/index#memory|GPU memory]]
- [[concepts/systems/batch-online-inference|Batch and online inference]]
- [[concepts/systems/inference-contract|Inference contract]]
- [[concepts/systems/inference-capacity-planning|Inference capacity planning]]
- [[concepts/systems/distributed-training-runbook|Distributed training runbook]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/systems/model-serving|Model serving]]
- [[infra/index|Infra]]
