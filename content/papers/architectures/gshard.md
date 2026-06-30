---
title: GShard
aliases:
  - papers/gshard
  - papers/scaling-giant-models-conditional-computation-automatic-sharding
tags:
  - papers
  - architectures
  - mixture-of-experts
  - systems
  - conditional-compute
---

# GShard

> GShard connects sparse MoE Transformer architecture with compiler-supported automatic sharding, making conditional computation practical at very large scale.

## Metadata

| Field | Value |
| --- | --- |
| Paper | GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding |
| Authors | Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, Zhifeng Chen |
| Year | 2020 preprint; 2021 conference |
| Venue | ICLR 2021 |
| arXiv | [2006.16668](https://arxiv.org/abs/2006.16668) |
| OpenReview | [qrwe7XHTmYb](https://openreview.net/forum?id=qrwe7XHTmYb) |
| Status | full note started |

## Question

[[papers/architectures/sparsely-gated-moe|Sparsely-Gated MoE]] shows how to increase model capacity with sparse expert activation. But a layer design alone does not solve the systems problem:

$$
\text{many experts}
\Rightarrow
\text{large parameter memory}
+
\text{routing}
+
\text{cross-device communication}.
$$

GShard asks:

$$
\text{Can sparse MoE Transformer models be scaled with minimal model-code changes using automatic sharding?}
$$

The answer combines a sparsely-gated MoE Transformer with compiler/runtime support for sharding tensor computation across many devices.

## Main Claim

GShard is both an architecture and systems paper. The reusable architecture claim is:

$$
\text{Transformer}
+
\text{sparse top-2 MoE FFN layers}
+
\text{automatic sharding}
\Rightarrow
\text{giant conditional-compute sequence models}.
$$

The systems claim is:

$$
\text{lightweight sharding annotations}
\rightarrow
\text{compiler-propagated partitioning}
\rightarrow
\text{large-scale distributed training}.
$$

## Architecture Contract

| Component | Contract |
| --- | --- |
| Base model | Transformer sequence-to-sequence model |
| Sparse layer | MoE feed-forward block inside Transformer |
| Router | token-level sparse gate over experts |
| Expert activation | top-2 style expert routing |
| Expert computation | selected feed-forward expert networks |
| Sharding layer | annotations plus XLA/compiler partitioning |
| Primary workload | multilingual neural machine translation |
| Main boundary | architecture quality and systems scalability are coupled |

The important modeling contract is:

$$
x_t
\rightarrow
\operatorname{router}(x_t)
\rightarrow
\{E_i(x_t)\}_{i\in S_t}
\rightarrow
y_t.
$$

The important systems contract is:

$$
\text{program}
+
\text{partition annotations}
\rightarrow
\text{sharded SPMD computation}.
$$

## MoE Transformer Layer

For token representation:

$$
x_t \in \mathbb{R}^{d},
$$

a dense Transformer FFN is:

$$
\operatorname{FFN}(x_t)
=
W_2\sigma(W_1x_t+b_1)+b_2.
$$

An MoE layer replaces that single FFN with expert networks:

$$
E_1,E_2,\dots,E_M.
$$

The router computes expert probabilities:

$$
p_t
=
\operatorname{softmax}(W_r x_t).
$$

The selected expert set is sparse:

$$
S_t=\operatorname{TopK}(p_t,2).
$$

The output is:

$$
\operatorname{MoE}(x_t)
=
\sum_{i\in S_t}
\alpha_{t,i}E_i(x_t),
$$

where $\alpha_{t,i}$ is the selected expert gate weight.

This keeps active expert compute small relative to total expert capacity:

$$
|\theta|_{\mathrm{active}}(x_t)
\ll
|\theta|_{\mathrm{total}}.
$$

## Why Automatic Sharding Matters

MoE creates a distribution problem. Expert weights and routed tokens must be partitioned across devices. If the user has to manually rewrite every tensor operation for a device mesh, the architecture is hard to use.

GShard's systems contribution is to let the model code stay close to ordinary TensorFlow/XLA style while adding sharding annotations to key tensors.

The conceptual flow is:

$$
\text{annotate selected tensors}
\rightarrow
\text{infer operator partitioning}
\rightarrow
\text{compile distributed program}.
$$

This is why GShard belongs in architecture reading even though it is systems-heavy: the architecture only works at the claimed scale if the sharding mechanism is part of the design.

## Capacity and Communication

Sparse experts change the bottleneck:

| Axis | Dense Transformer | GShard MoE Transformer |
| --- | --- | --- |
| FFN capacity | one dense FFN per layer | many expert FFNs per sparse layer |
| active experts | all dense FFN weights used | only selected experts used per token |
| parameter memory | dense layer parameters | expert pool distributed across devices |
| communication | mostly parallel training collectives | routing, dispatch, combine, and sharded expert computation |
| scaling risk | compute/memory grows directly | load balance and communication can dominate |

The claim is not simply "more parameters are better." The claim is:

$$
\text{sparse activation}
+
\text{device-aware sharding}
\Rightarrow
\text{practical large-capacity training}.
$$

## Relation to MoE Lineage

| Paper | Main Role |
| --- | --- |
| [Sparsely-Gated MoE](/papers/architectures/sparsely-gated-moe) | defines sparse expert routing as a reusable neural layer |
| GShard | scales MoE Transformer with automatic sharding |
| [Switch Transformer](/papers/architectures/switch-transformer) | simplifies routing to top-1 for sparse Transformer scaling |
| [GLaM](/papers/architectures/glam) | applies sparse expert scaling to large autoregressive language models |

The reading order is:

$$
\text{MoE layer}
\rightarrow
\text{sharded MoE Transformer}
\rightarrow
\text{simplified sparse Transformer}
\rightarrow
\text{sparse expert LLM}.
$$

## Evidence

| Evidence Type | What It Supports |
| --- | --- |
| multilingual translation experiments | sparse MoE Transformer can improve large-scale sequence-to-sequence quality |
| scaling measurements | GShard can train very large MoE models across many accelerators |
| sharding API design | model developers can express partitioning with relatively light annotations |
| efficiency and memory measurements | systems constraints are central to the architecture's feasibility |

## Evaluation Risks

- Architecture gains are entangled with data scale, translation task setup, accelerator count, and compiler/runtime implementation.
- Total parameter count should not be read as active per-token compute.
- A systems result on a large TPU setup may not transfer to smaller or different hardware.
- Sparse expert routing introduces load balance and communication costs that simple FLOP estimates can hide.
- Translation quality claims should be read with task, language-pair, data, and baseline boundaries.

## Why It Belongs in Architecture Papers

GShard changes how an architecture can be scaled:

$$
\text{model architecture}
\not\perp
\text{compiler and distributed execution}.
$$

For this wiki, GShard is a canonical paper for reading sparse expert architectures when the real question is not just "what is the layer?" but:

- how are experts routed;
- how are experts distributed;
- what computation is active per token;
- where does communication happen;
- which claims are architecture claims versus systems claims?

## Concepts

- [[concepts/architectures/mixture-of-experts|Mixture of Experts]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
- [[concepts/systems/distributed-training-runbook|Distributed training]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]

## Related

- [[papers/architectures/sparsely-gated-moe|Sparsely-Gated MoE]]
- [[papers/architectures/switch-transformer|Switch Transformer]]
- [[papers/architectures/glam|GLaM]]
- [[papers/architectures/t5|T5]]
