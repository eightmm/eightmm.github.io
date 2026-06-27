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

## Diagnostic Map

- [[infra/hardware/memory-hierarchy|Memory hierarchy]]
- [[infra/gpu/index#bottleneck-taxonomy|Bottleneck taxonomy]]
- [[infra/gpu/index#memory|Memory]]
- [[infra/gpu/index#utilization|Utilization]]
- [[infra/gpu/index#driver-and-cuda|Driver and CUDA]]

## Bottleneck Taxonomy

A GPU bottleneck taxonomy separates capacity, bandwidth, compute, input pipeline, synchronization, communication, and scheduler limits. It prevents "GPU utilization is low" from becoming a vague diagnosis.

A training or inference step can be decomposed as:

$$
t_{\mathrm{step}}
=
t_{\mathrm{input}}
+ t_{\mathrm{host}}
+ t_{\mathrm{transfer}}
+ t_{\mathrm{gpu}}
+ t_{\mathrm{sync}}
+ t_{\mathrm{comm}}
$$

| Type | Symptom | Typical evidence |
| --- | --- | --- |
| Capacity | Out-of-memory or tiny batch size | Memory allocation, batch/context sensitivity |
| Bandwidth | High memory traffic, weak FLOP utilization | Profiler memory bandwidth, tensor shape pattern |
| Compute | Kernels dominate wall time | High GPU utilization and stable input pipeline |
| Input pipeline | GPU waits for data | CPU usage, data-loader timing, storage IO |
| Synchronization | Frequent stalls | Host sync calls, logging, metric aggregation |
| Communication | Poor multi-GPU scaling | All-reduce time, interconnect saturation |
| Scheduler | Long wait, preemption, or poor allocation fit | Queue time, walltime, resource request mismatch |

## Memory

GPU memory is usually the first hard limit in training and inference. A run can be compute-light but still fail if weights, activations, optimizer state, gradients, or KV cache exceed device memory.

For training, a rough memory decomposition is:

$$
M_{\mathrm{total}}
\approx
M_{\mathrm{weights}}
+ M_{\mathrm{gradients}}
+ M_{\mathrm{optimizer}}
+ M_{\mathrm{activations}}
+ M_{\mathrm{buffers}}
$$

For autoregressive inference, KV cache often dominates:

$$
M_{\mathrm{KV}}
\approx
2 \cdot L \cdot H \cdot T \cdot d_{\mathrm{head}} \cdot b
$$

where $L$ is number of layers, $H$ is number of attention heads, $T$ is context length, and $b$ is bytes per value.

Key checks:

- Is memory used by parameters, activations, optimizer state, batch size, or KV cache?
- Does memory grow every step, suggesting graph retention or logging leakage?
- Would mixed precision, gradient checkpointing, smaller batch size, sharded optimizer, or shorter context solve the bottleneck?
- Is memory the primary bottleneck, or only a symptom of input, synchronization, or communication stalls?

## Utilization

GPU utilization is a symptom, not a complete diagnosis. Low utilization can come from data loading, CPU preprocessing, synchronization, communication, small batch sizes, memory stalls, or simply a workload that is not compute-heavy.

A simple utilization view is:

$$
u
\approx
\frac{t_{\mathrm{gpu\ work}}}
{t_{\mathrm{wall}}}
$$

High utilization alone does not prove good throughput. The useful metric is work completed per second under a fixed quality and memory budget.

Practical diagnosis:

1. Measure step time, GPU utilization, GPU memory, CPU usage, and I/O wait.
2. Check whether data loading keeps up with compute.
3. Check batch size, precision, and tensor shapes.
4. Check synchronization points and distributed communication.
5. Profile a short representative run before changing code.

## Driver and CUDA

Most "GPU not working" failures on a shared machine come from a mismatch between the NVIDIA driver, the CUDA runtime a framework was built against, and the toolkit on the host. The driver caps the maximum CUDA version; a newer runtime than the driver supports will fail to initialize.

The compatibility chain is:

$$
\text{kernel driver}
\rightarrow
\text{CUDA runtime}
\rightarrow
\text{framework build}
\rightarrow
\text{application kernel}
$$

Debug from the host layer upward instead of changing project dependencies first.

Practical checks:

- Confirm the driver sees the GPUs before blaming the framework.
- Check the framework's bundled CUDA runtime, not just the system toolkit.
- Treat the driver's reported CUDA version as a ceiling for runtime compatibility.
- After a driver upgrade, reboot or reload the kernel module before testing.
- Reproduce in a clean environment to separate a host problem from a project problem.
- Record the error class and compatibility layer, not private node names or project paths.

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
- Classify the bottleneck with [[infra/gpu/index#bottleneck-taxonomy|GPU bottleneck taxonomy]] before changing model code.
- Keep public notes generic; do not publish private hosts or allocations.

## Related

- [[concepts/systems/distributed-training-runbook|Distributed training]]
- [[infra/hpc/distributed-training|Distributed training on HPC]]
- [[infra/hardware/memory-hierarchy|Memory hierarchy]]
- [[concepts/systems/inference-serving|Inference serving]]
- [[infra/hpc/slurm|Slurm]]
- [[infra/io/data-loading|Data loading and IO]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[infra/index|Infra]]
