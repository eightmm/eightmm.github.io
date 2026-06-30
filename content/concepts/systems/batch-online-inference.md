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

## Mode Contract

| Field | Batch inference | Online inference |
| --- | --- | --- |
| unit | dataset row, file, molecule, document, shard | request |
| objective | throughput, reproducibility, cost | latency, availability, UX |
| failure handling | retry item or shard | timeout, fallback, clear error |
| ordering | often reorderable | request order may matter |
| logging | artifact manifest and run id | request log with privacy boundary |
| versioning | model + data + output version | model + endpoint + client contract |

The same model can be correct in batch mode and unsuitable online if preprocessing, latency, memory, or error handling differs.

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

## Dynamic Batching

Online systems often batch requests briefly to improve GPU utilization:

$$
B_t
=
\{x_j: t_j \in [t, t+\Delta]\}
$$

This improves throughput when model execution benefits from batching, but increases waiting time by up to the batching window $\Delta$.

| Choice | Effect |
| --- | --- |
| larger batch | better hardware utilization, higher latency |
| shorter timeout | better latency, more partial batches |
| queue priority | protects interactive users or small requests |
| max sequence length | controls memory and tail latency |

## Reproducibility Boundary

Batch inference can usually produce a manifest:

$$
\text{output}
=
(\text{input version},\ \text{model version},\ \text{config},\ \text{shard},\ \text{timestamp})
$$

Online inference is harder to reproduce because requests arrive from a live distribution and logs may be redacted. Public notes should distinguish `same model` from `same operational behavior`.

## Checks

- Can failed batch items be retried without duplicating successful outputs?
- Is online timeout behavior documented?
- Does batching change numerical results or output ordering?
- Are batch outputs tied to input/model/preprocessing versions?
- Are request logs privacy-safe and sufficient for debugging?

## Related

- [[concepts/systems/inference|Inference]]
- [[concepts/systems/model-serving|Model serving]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[concepts/systems/inference-serving|Inference serving]]
- [[concepts/systems/storage-io|Storage and IO]]
