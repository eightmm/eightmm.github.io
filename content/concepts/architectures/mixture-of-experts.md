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
