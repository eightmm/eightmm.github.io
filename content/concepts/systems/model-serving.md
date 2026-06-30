---
title: Model Serving
tags:
  - systems
  - serving
  - inference
---

# Model Serving

Model serving turns inference into an interface that users, workflows, or other systems can call reliably. Serving concerns include request batching, model loading, resource isolation, timeouts, monitoring, versioning, and rollout policy.

The serving path is usually:

$$
\text{request}
\rightarrow \text{preprocess}
\rightarrow \text{model inference}
\rightarrow \text{postprocess}
\rightarrow \text{response}
$$

## Serving Contract

Serving should expose a stable contract, not just a model checkpoint.

$$
S
=
(\text{input schema},\ \text{preprocess},\ \text{model version},\ \text{postprocess},\ \text{SLO},\ \text{logging})
$$

| Contract field | Question |
| --- | --- |
| input schema | what is accepted and rejected? |
| preprocessing | which transforms and versions run before inference? |
| model version | which weights/config/tokenizer or featurizer are active? |
| postprocessing | thresholds, calibration, formatting, validation |
| SLO | latency, throughput, availability, timeout, error budget |
| logging | what is recorded, redacted, or never stored? |

The model version and preprocessing version must move together. A serving bug can come from a schema or preprocessing mismatch even if weights are unchanged.

## Serving Modes

| Mode | Use when | Main risk |
| --- | --- | --- |
| synchronous API | user waits for response | tail latency and timeout |
| async job | request can complete later | status tracking and retries |
| streaming | partial output useful | cancellation and partial validity |
| batch endpoint | many examples scored together | output versioning and retry shards |
| agent tool | LLM calls service as action | side effects and contract validation |

## Observability

Serving metrics should separate model behavior from system behavior.

| Signal | Tells you |
| --- | --- |
| request count and error rate | interface health |
| p50/p95/p99 latency | tail behavior |
| queue length | capacity pressure |
| batch size distribution | batching effectiveness |
| input rejection rate | schema or client mismatch |
| model confidence/calibration drift | prediction distribution change |
| resource utilization | CPU/GPU/memory bottleneck |

## Key Decisions

- Single model or multiple model versions.
- Synchronous, asynchronous, batch, or streaming interface.
- Static batching, dynamic batching, or continuous batching.
- CPU preprocessing and GPU inference boundary.
- Canary, shadow, blue-green, or batch rollout strategy.
- Timeout, retry, and fallback policy.
- Logging policy that avoids private data capture.

## Checks

- What is the public contract of the endpoint or workflow?
- Does the endpoint follow a documented [[concepts/systems/inference-contract|inference contract]]?
- Are model version and preprocessing version tied together?
- Is the deployment strategy explicit and reversible?
- Are tail latency and error rate measured?
- Can the service reject unsupported inputs clearly?
- Does logging exclude secrets, private inputs, and unpublished data?
- Are preprocessing, model, and postprocessing versions deployed atomically?
- Are model-quality metrics separated from system-health metrics?

## Related

- [[concepts/systems/inference-contract|Inference contract]]
- [[concepts/systems/model-card|Model card]]
- [[concepts/systems/model-versioning|Model versioning]]
- [[concepts/systems/deployment-strategy|Deployment strategy]]
- [[concepts/systems/inference|Inference]]
- [[concepts/systems/batch-online-inference|Batch and online inference]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[concepts/systems/failure-recovery|Failure recovery]]
- [[concepts/systems/observability|Observability]]
- [[concepts/systems/inference-serving|Inference serving]]
- [[concepts/systems/inference-capacity-planning|Inference capacity planning]]
- [[infra/server-ops/monitoring|Monitoring shared machines]]
- [[agents/tools/tool-use|Tool use]]
