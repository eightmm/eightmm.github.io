---
title: Mixture of Experts
tags:
  - architectures
  - mixture-of-experts
  - routing
---

# Mixture of Experts

Mixture-of-experts models route each input through a subset of expert modules. The pattern separates model capacity from the amount of computation used per token or example.

A sparse MoE layer can be summarized as:

$$
r(x)=\operatorname{softmax}(W_r x)
$$

$$
S(x)=\operatorname{TopK}(r(x))
$$

$$
y = \sum_{m \in S(x)}
g_m(x) E_m(x)
$$

Here $r(x)$ is the router distribution, $S(x)$ is the selected expert set, $E_m$ is expert $m$, and $g_m(x)$ is the normalized routing weight for selected experts.

Sparse routing separates total parameters from activated parameters:

$$
|\theta|_{\mathrm{total}}
\gg
|\theta|_{\mathrm{active}}(x)
$$

This is the main reason MoE is attractive for scaling, but it also makes serving and load balancing harder.

## Key Ideas

- A router selects one or more experts for each token, example, node, or region.
- Sparse MoE increases parameter count while activating only a subset of parameters per input.
- Experts are often feed-forward blocks, but the pattern can apply to other modules.
- Load-balancing losses or routing constraints keep a few experts from receiving all traffic.
- Routing can create specialization, but specialization should be checked rather than assumed.

## Router Checks

| Field | Why It Matters |
| --- | --- |
| Granularity | token, sequence, graph, modality, or task routing changes the claim |
| Top-k | controls active compute and redundancy |
| Capacity factor | determines whether tokens are dropped or rerouted under load |
| Auxiliary loss | prevents expert collapse or severe imbalance |
| Expert type | usually FFN, but can be modality-specific or task-specific |
| Serving path | sparse routing can hurt latency even when FLOPs look low |

## Load Balancing

Sparse routing creates a systems problem: the model can have enough total experts while a few experts receive most tokens. A simple routing utilization for expert $m$ is:

$$
u_m
= \frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}[m \in S(x_i)]
$$

Balanced routing wants $u_m$ to be close to $1/M$ for $M$ experts, while still sending each token to useful experts. Many MoE papers add an auxiliary loss that penalizes imbalance:

$$
\mathcal{L}
= \mathcal{L}_{\mathrm{task}}
+ \alpha \mathcal{L}_{\mathrm{balance}}
$$

The exact form varies, so paper notes should record router probability, selected expert frequency, capacity factor, and dropped-token policy separately.

## Capacity and Dropped Tokens

Each expert often has a capacity limit per batch:

$$
\text{capacity}
= \left\lceil
\frac{N \cdot k}{M}
\cdot c
\right\rceil
$$

where $N$ is tokens, $k$ is top-k routing, $M$ is number of experts, and $c$ is the capacity factor. If too many tokens route to one expert, implementations may drop, reroute, or pad tokens. That detail changes both quality and latency.

## MoE Claim Types

| Claim | Evidence Needed |
| --- | --- |
| more capacity | total parameters and active parameters reported separately |
| cheaper compute | activated FLOPs, memory traffic, communication, and wall time |
| expert specialization | routing distribution and per-task/per-domain behavior |
| scalable training | expert parallelism, all-to-all overhead, load balance |
| efficient serving | latency at realistic batch size, cache behavior, and routing overhead |

## Canonical Papers

| Paper | Why It Matters |
| --- | --- |
| [Sparsely-Gated MoE](/papers/architectures/sparsely-gated-moe) | defines sparse expert routing, noisy top-$k$ gates, and load-balancing as part of the layer contract |
| [Switch Transformer](/papers/architectures/switch-transformer) | simplifies sparse Transformer MoE with top-1 routing |
| [GLaM](/papers/architectures/glam) | applies sparse expert routing to large autoregressive language-model scaling |

## Where MoE Fits

MoE is not a replacement for choosing a base architecture. It is usually a routing layer inserted into a Transformer, MLP block, multimodal system, or task-specific model.

| Base context | What MoE changes |
| --- | --- |
| Transformer FFN | increases parameter capacity per token with sparse FFN activation |
| multimodal model | routes by modality, task, or token type |
| multi-task model | separates capacity across tasks while sharing interface layers |
| agent system | should be distinguished from explicit tool routing or workflow routing |

If the router chooses external tools, APIs, or actions, the note belongs closer to [[agents/tools/tool-use|tool use]] than to MoE.

## Practical Checks

- Check routing granularity: token-level, sequence-level, graph-level, modality-level, or task-level.
- Track top-k choice, capacity limits, dropped tokens, and auxiliary routing losses.
- Compare parameter count, activated parameter count, memory traffic, and inference latency separately.
- For agent or tool systems, distinguish MoE routing from explicit [[agents/tools/tool-use|tool use]].
- Do not infer interpretability from expert names; measure routing distributions and task behavior.

## Related

- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/mlp|MLP]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[agents/index|Agents]]
- [[concepts/learning/index|Learning methods]]
