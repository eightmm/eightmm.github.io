---
title: Inference Serving
unlisted: true
tags:
  - infra
  - inference
---

# Inference Serving

Serving is mainly an AI systems concept, not an infra category by itself. Start from [[concepts/systems/inference-serving|Inference serving]] and use infra notes only when the bottleneck is hardware or operations.

| Question | Go To |
| --- | --- |
| What is serving and batching? | [Inference serving](/concepts/systems/inference-serving) |
| What contract should the model expose? | [Inference contract](/concepts/systems/inference-contract) |
| Is this latency or throughput limited? | [Latency and throughput](/concepts/systems/latency-throughput) |
| Is the issue GPU memory or utilization? | [GPU](/infra/gpu/) |

Do not publish private endpoints, hostnames, ports, request traces, logs, prompts, dashboards, or credentials in public serving notes.
