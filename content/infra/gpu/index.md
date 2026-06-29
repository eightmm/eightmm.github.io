---
title: GPU
aliases:
  - infra/gpu
tags:
  - infra
  - gpu
---

# GPU

GPU는 training과 inference 뒤의 dense linear algebra를 가속합니다. 반복해서 문제가 되는 제약은 raw FLOPs만이 아니라 memory capacity, memory bandwidth, interconnect입니다.

공개 research-engineering note에서는 GPU 작업을 resource diagnosis 문제로 봅니다.

$$
\text{throughput}
=
f(\text{model}, \text{batch}, \text{precision}, \text{memory}, \text{input}, \text{communication})
$$

유용한 질문은 현재 workflow를 제한하는 항이 무엇인가입니다.

## 진단 지도

- [[infra/hardware/memory-hierarchy|Memory hierarchy]]
- [[infra/gpu/index#bottleneck-taxonomy|Bottleneck taxonomy]]
- [[infra/gpu/index#memory|Memory]]
- [[infra/gpu/index#utilization|Utilization]]
- [[infra/gpu/index#driver-and-cuda|Driver and CUDA]]

## Bottleneck taxonomy

GPU bottleneck taxonomy는 capacity, bandwidth, compute, input pipeline, synchronization, communication, scheduler limit를 분리합니다. 이렇게 해야 “GPU utilization이 낮다”는 말이 모호한 진단으로 끝나지 않습니다.

Training 또는 inference step은 아래처럼 분해할 수 있습니다.

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

| 유형 | 증상 | 일반적인 evidence |
| --- | --- | --- |
| Capacity | OOM 또는 너무 작은 batch size | memory allocation, batch/context sensitivity |
| Bandwidth | 높은 memory traffic, 낮은 FLOP utilization | profiler memory bandwidth, tensor shape pattern |
| Compute | kernel이 wall time을 지배 | 높은 GPU utilization과 안정적인 input pipeline |
| Input pipeline | GPU가 data를 기다림 | CPU usage, data-loader timing, storage IO |
| Synchronization | 잦은 stall | host sync call, logging, metric aggregation |
| Communication | multi-GPU scaling 저하 | all-reduce time, interconnect saturation |
| Scheduler | 긴 wait, preemption, allocation mismatch | queue time, walltime, resource request mismatch |

## Memory

GPU memory는 training과 inference에서 보통 가장 먼저 만나는 hard limit입니다. Workload가 compute-light하더라도 weight, activation, optimizer state, gradient, KV cache가 device memory를 넘으면 실패합니다.

Training memory는 대략 아래처럼 분해할 수 있습니다.

$$
M_{\mathrm{total}}
\approx
M_{\mathrm{weights}}
+ M_{\mathrm{gradients}}
+ M_{\mathrm{optimizer}}
+ M_{\mathrm{activations}}
+ M_{\mathrm{buffers}}
$$

Autoregressive inference에서는 KV cache가 지배적인 경우가 많습니다.

$$
M_{\mathrm{KV}}
\approx
2 \cdot L \cdot H \cdot T \cdot d_{\mathrm{head}} \cdot b
$$

여기서 $L$은 layer 수, $H$는 attention head 수, $T$는 context length, $b$는 value 하나의 byte 수입니다.

핵심 check:

- memory가 parameter, activation, optimizer state, batch size, KV cache 중 어디에 쓰이는가?
- 매 step memory가 증가해서 graph retention이나 logging leakage를 의심해야 하는가?
- mixed precision, gradient checkpointing, smaller batch size, sharded optimizer, shorter context가 bottleneck을 해결하는가?
- memory가 primary bottleneck인가, 아니면 input, synchronization, communication stall의 증상인가?

## Utilization

GPU utilization은 증상이지 완전한 진단이 아닙니다. Low utilization은 data loading, CPU preprocessing, synchronization, communication, small batch size, memory stall, 또는 compute-heavy하지 않은 workload에서 나올 수 있습니다.

단순한 utilization 관점은 아래와 같습니다.

$$
u
\approx
\frac{t_{\mathrm{gpu\ work}}}
{t_{\mathrm{wall}}}
$$

High utilization만으로 좋은 throughput이 증명되지는 않습니다. 유용한 metric은 고정된 quality와 memory budget 아래에서 초당 완료한 work입니다.

실전 진단:

1. step time, GPU utilization, GPU memory, CPU usage, I/O wait를 측정합니다.
2. data loading이 compute를 따라오는지 확인합니다.
3. batch size, precision, tensor shape를 확인합니다.
4. synchronization point와 distributed communication을 확인합니다.
5. code를 바꾸기 전에 짧고 대표적인 run을 profile합니다.

## Driver and CUDA

Shared machine에서 “GPU가 안 된다”는 실패는 대부분 NVIDIA driver, framework가 빌드된 CUDA runtime, host toolkit 사이의 mismatch에서 나옵니다. Driver는 지원 가능한 최대 CUDA version을 제한하고, driver가 지원하지 않는 더 새로운 runtime은 initialize에 실패합니다.

Compatibility chain은 아래와 같습니다.

$$
\text{kernel driver}
\rightarrow
\text{CUDA runtime}
\rightarrow
\text{framework build}
\rightarrow
\text{application kernel}
$$

처음부터 project dependency를 바꾸기보다 host layer에서 위로 올라가며 debug합니다.

실전 check:

- framework를 의심하기 전에 driver가 GPU를 보는지 확인합니다.
- system toolkit만 보지 말고 framework에 bundled된 CUDA runtime을 확인합니다.
- driver가 보고하는 CUDA version은 runtime compatibility의 상한으로 봅니다.
- driver upgrade 뒤에는 test 전에 reboot하거나 kernel module을 reload합니다.
- clean environment에서 재현해 host 문제와 project 문제를 분리합니다.
- private node name이나 project path 대신 error class와 compatibility layer를 기록합니다.

## 진단 축

- Capacity: model, activation, optimizer state, cache가 들어가는가?
- Bandwidth: kernel이 arithmetic보다 memory movement에 제한되는가?
- Compute: tensor core 또는 GPU kernel이 wall time을 지배하는가?
- Input pipeline: CPU preprocessing이나 storage가 GPU를 굶기는가?
- Communication: multi-GPU synchronization이 scaling을 지배하는가?
- Scheduler: resource request나 walltime이 code보다 run을 더 크게 좌우하는가?

## First Measurements

GPU note는 추측보다 작은 측정값에서 시작합니다.

| 측정값 | 이유 |
| --- | --- |
| step time | throughput 변화의 기본 단위 |
| allocated / reserved memory | true capacity pressure와 fragmentation 구분 |
| GPU utilization | compute work가 있는지 보는 rough signal |
| dataloader wait | input pipeline starvation 확인 |
| host CPU and RAM | preprocessing, pin memory, worker pressure 확인 |
| communication time | distributed scaling 병목 확인 |

## 실전 check

- model weight, activation, optimizer state, KV cache의 memory headroom을 추적합니다.
- precision을 hardware와 numerical tolerance에 맞춥니다.
- optimize 전에 profile합니다.
- model code를 바꾸기 전에 [[infra/gpu/index#bottleneck-taxonomy|GPU bottleneck taxonomy]]로 bottleneck을 분류합니다.
- public note는 generic하게 유지하고 private host나 allocation을 공개하지 않습니다.

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
