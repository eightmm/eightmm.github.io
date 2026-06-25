---
title: Batch and Online Inference
tags:
  - systems
  - inference
  - serving
---

# Batch and Online Inference

Batch and online inference are different operational modes for using a trained model. They often use the same model weights but optimize for different constraints.

Batch inference processes a dataset:

$$
\hat{Y}
=
\{f_{\hat{\theta}}(x_i)\}_{i=1}^{N}
$$

Online inference processes requests as they arrive:

$$
\hat{y}_j
=
f_{\hat{\theta}}(x_j),
\qquad
x_j \sim p_{\mathrm{request}}(x)
$$

## Batch Inference

Batch mode is optimized for throughput, cost, and reproducibility. It is common for dataset scoring, embedding generation, virtual screening, offline evaluation, and paper-processing pipelines.

Key questions:

- Can examples be reordered?
- Can failed items be retried independently?
- Is output versioning tied to model and data version?
- Is the bottleneck GPU compute, CPU preprocessing, storage IO, or network transfer?

## Online Inference

Online mode is optimized for latency, reliability, and request contract. It is common for APIs, interactive tools, agent loops, dashboards, and user-facing applications.

Key questions:

- What are latency and timeout targets?
- Can requests be batched without changing behavior?
- How are invalid inputs rejected?
- What is logged, redacted, or never stored?

## Tradeoff

Throughput is roughly:

$$
\operatorname{throughput}
=
\frac{\text{completed examples}}{\text{wall-clock time}}
$$

Latency is per request:

$$
\operatorname{latency}_j
=
t_{\mathrm{response},j}
-
t_{\mathrm{arrival},j}
$$

Optimizing one can hurt the other.

## Related

- [[concepts/systems/inference|Inference]]
- [[concepts/systems/model-serving|Model serving]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[infra/inference-serving|Inference serving]]
- [[concepts/systems/storage-io|Storage and IO]]
