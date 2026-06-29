---
title: Inference Serving
unlisted: true
tags:
  - infra
  - inference
---

# Inference Serving

Serving is mainly an AI systems concept, not an infra category by itself. Start from [[concepts/systems/inference-serving|Inference serving]] and use infra notes only when the bottleneck is hardware or operations.

An infra-facing serving note should keep the serving path explicit:

$$
\text{request}
\rightarrow
\text{queue}
\rightarrow
\text{batch}
\rightarrow
\text{model runtime}
\rightarrow
\text{postprocess}
\rightarrow
\text{response}
$$

| Question | Go To |
| --- | --- |
| What is serving and batching? | [Inference serving](/concepts/systems/inference-serving) |
| What contract should the model expose? | [Inference contract](/concepts/systems/inference-contract) |
| Is this latency or throughput limited? | [Latency and throughput](/concepts/systems/latency-throughput) |
| Is the issue GPU memory or utilization? | [GPU](/infra/gpu/) |

## Operational Checks

| Check | Why |
| --- | --- |
| Request contract | input size, output size, timeout, and failure mode shape capacity |
| Queueing policy | latency can be dominated before the model starts |
| Batching policy | throughput improves only if padding and tail latency stay controlled |
| Runtime placement | CPU preprocessing, GPU compute, storage, and network may bottleneck separately |
| Observability | logs and metrics must be sanitized before becoming public examples |

## Route

| Strongest topic | Put it in |
| --- | --- |
| batching, decoding, cache behavior, model I/O contract | [AI Systems](/ai/systems), [Inference serving](/concepts/systems/inference-serving) |
| GPU memory, node sizing, runtime failure, queue behavior | [Infra](/infra), [GPU](/infra/gpu/) |
| tool-using LLM workflow around a served model | [Agents](/agents) |

Do not publish private endpoints, hostnames, ports, request traces, logs, prompts, dashboards, or credentials in public serving notes.
