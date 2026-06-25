---
title: Model Serving
tags:
  - systems
  - serving
  - inference
---

# Model Serving

Model serving turns inference into an interface that users, workflows, or other systems can call reliably. Serving concerns include request batching, model loading, resource isolation, timeouts, monitoring, and versioning.

The serving path is usually:

$$
\text{request}
\rightarrow \text{preprocess}
\rightarrow \text{model inference}
\rightarrow \text{postprocess}
\rightarrow \text{response}
$$

## Key Decisions

- Single model or multiple model versions.
- Synchronous, asynchronous, batch, or streaming interface.
- Static batching, dynamic batching, or continuous batching.
- CPU preprocessing and GPU inference boundary.
- Timeout, retry, and fallback policy.
- Logging policy that avoids private data capture.

## Checks

- What is the public contract of the endpoint or workflow?
- Are model version and preprocessing version tied together?
- Are tail latency and error rate measured?
- Can the service reject unsupported inputs clearly?
- Does logging exclude secrets, private inputs, and unpublished data?

## Related

- [[concepts/systems/inference|Inference]]
- [[concepts/systems/batch-online-inference|Batch and online inference]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[concepts/systems/failure-recovery|Failure recovery]]
- [[infra/inference-serving|Inference serving]]
- [[infra/server-ops/monitoring|Monitoring shared machines]]
- [[agents/tools/tool-use|Tool use]]
