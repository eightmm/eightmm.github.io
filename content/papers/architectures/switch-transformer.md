---
title: Switch Transformer
aliases:
  - papers/switch-transformer
  - papers/switch-transformers
tags:
  - papers
  - architectures
  - mixture-of-experts
  - routing
---

# Switch Transformer

> The paper simplified sparse mixture-of-experts routing by sending each token to a single expert.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity |
| Authors | William Fedus, Barret Zoph, Noam Shazeer |
| Year | 2021 preprint; 2022 journal |
| Venue | JMLR 2022 |
| arXiv | [2101.03961](https://arxiv.org/abs/2101.03961) |
| JMLR | [23(120):1-39](https://jmlr.org/papers/v23/21-0998.html) |
| Status | verified |

## Question

Dense models reuse the same parameters for every token. Sparse mixture-of-experts increases capacity with conditional computation, but routing, communication, and instability make MoE systems hard to train. The question was whether routing could be simplified enough to make sparse scaling practical.

## Main Claim

Top-1 expert routing can simplify MoE training while increasing parameter count without increasing per-token computation proportionally.

Narrowed claim:

$$
y_t
=
E_{r(t)}(x_t)
$$

where each token $x_t$ is routed to one selected expert $E_{r(t)}$ rather than a dense combination of all experts.

## Method

Switch Transformer replaces the feed-forward sublayer in a Transformer block with a sparse expert layer.

| Component | Role |
| --- | --- |
| router | maps each token to one expert |
| expert FFN | processes only assigned tokens |
| capacity factor | limits how many tokens each expert receives |
| auxiliary load-balancing loss | discourages expert collapse |

The main architecture idea is conditional computation:

$$
\text{parameters used per token}
\ll
\text{total parameters}
$$

## Evidence

| Claim | Evidence in paper | Caveat |
| --- | --- | --- |
| Sparse expert routing can improve scaling efficiency | pretraining speed and quality comparisons against T5-style dense baselines | results are tied to language-model training setup |
| Top-1 routing is simpler than earlier sparse MoE routing | routing algorithm and systems measurements | simplicity does not remove distributed-systems complexity |
| Sparse models can be trained at very large parameter counts | trillion-parameter scale experiments | scale claims depend on hardware and data pipeline |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task | language-model pretraining and transfer |
| Architecture family | sparse Mixture of Experts Transformer |
| Main comparison | dense T5-style Transformer baselines |
| Main metric family | pretraining loss, downstream performance, speed, parameter scale |
| Not directly tested | general MoE behavior in all modalities or small-data settings |

## Limitations

- Sparse parameter count is not the same as per-token compute; communication and routing overhead matter.
- Expert imbalance, token dropping, and routing instability can affect training quality.
- The architecture is coupled to distributed training systems.
- The paper is important for architecture and systems scaling, but its empirical evidence is centered on language modeling.

## Why It Matters

Switch Transformer is an anchor paper for conditional computation and sparse expert routing. It belongs in architecture papers because the lasting contribution is a model block and scaling route, even though the main experiments are language-model based.

## Connections

- [[concepts/architectures/mixture-of-experts|Mixture of experts]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/systems/distributed-training-runbook|Distributed training]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/index|Architecture papers]]
