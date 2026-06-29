---
title: Inference Capacity Planning
unlisted: true
tags:
  - infra
  - inference
---

# Inference Capacity Planning

Capacity planning sits between model behavior and hardware limits. Start from [[concepts/systems/inference-capacity-planning|Inference capacity planning]] for the reusable systems concept; infra notes provide the resource mental model.

The basic capacity question is:

$$
\lambda_{\max}
\approx
\frac{N_{\mathrm{replicas}} \cdot B_{\mathrm{eff}}}{T_{\mathrm{batch}}}
$$

where $\lambda_{\max}$ is sustainable request rate, $N_{\mathrm{replicas}}$ is the number of serving replicas, $B_{\mathrm{eff}}$ is effective batch size after padding and scheduling overhead, and $T_{\mathrm{batch}}$ is end-to-end batch service time.

| Question | Go To |
| --- | --- |
| How many requests fit under latency and memory limits? | [Inference capacity planning](/concepts/systems/inference-capacity-planning) |
| What is the serving path? | [Inference serving](/concepts/systems/inference-serving) |
| What hardware limit matters? | [Hardware](/infra/hardware/), [GPU](/infra/gpu/) |
| How should results be recorded? | [Reproducible run record](/infra/reproducibility/run-record) |

## Planning Variables

| Variable | Meaning | Common trap |
| --- | --- | --- |
| request shape | prompt length, output length, modality size, batchability | average case hides tail latency |
| memory budget | weights, KV cache, activations, workspace, fragmentation | only counting model weights |
| service time | queueing, prefill, decode, postprocess, network | measuring kernel time only |
| concurrency | replicas, workers, streams, batch scheduler | increasing concurrency until latency collapses |
| utilization | GPU compute, memory bandwidth, CPU, storage, network | assuming GPU utilization is the only bottleneck |

## Boundary

| Infra question | AI systems question |
| --- | --- |
| How many GPUs, workers, queues, and replicas are needed? | What is the model contract and serving behavior? |
| Which resource saturates first? | Which decoding, batching, or caching policy is used? |
| What is the public run record? | What metric supports the user-facing claim? |

Public capacity notes should use generic workloads and formulas. Do not publish real traffic, private utilization, endpoint names, account names, dashboards, or unpublished benchmark results.
