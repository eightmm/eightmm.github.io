---
title: GPU
aliases:
  - infra/gpu
tags:
  - infra
  - gpu
---

# GPU

GPUs accelerate the dense linear algebra behind training and inference. The recurring constraints are memory capacity, memory bandwidth, and interconnect, not only raw FLOPs.

For public research-engineering notes, treat GPU work as a resource diagnosis problem:

$$
\text{throughput}
=
f(\text{model}, \text{batch}, \text{precision}, \text{memory}, \text{input}, \text{communication})
$$

The useful question is which term limits the current workflow.

## Notes

- [[infra/gpu/bottleneck-taxonomy|GPU bottleneck taxonomy]]
- [[infra/gpu/utilization|GPU utilization]]
- [[infra/gpu/memory|GPU memory]]
- [[infra/server-ops/gpu-driver-cuda|GPU driver and CUDA debugging]]

## Diagnostic Axes

- Capacity: can the model, activations, optimizer state, and cache fit?
- Bandwidth: are kernels limited by memory movement rather than arithmetic?
- Compute: are tensor cores or GPU kernels the dominant wall time?
- Input pipeline: is CPU preprocessing or storage starving the GPU?
- Communication: does multi-GPU synchronization dominate scaling?
- Scheduler: does resource request or walltime shape the run more than code?

## Practical Checks

- Track memory headroom: model weights, activations, optimizer state, and KV cache.
- Match precision to the hardware and numerical tolerance.
- Profile before optimizing.
- Classify the bottleneck with [[infra/gpu/bottleneck-taxonomy|GPU bottleneck taxonomy]] before changing model code.
- Keep public notes generic; do not publish private hosts or allocations.

## Related

- [[infra/training/distributed-training|Distributed training]]
- [[infra/inference/serving|Inference serving]]
- [[infra/hpc/slurm|Slurm]]
- [[infra/index|Infra]]
