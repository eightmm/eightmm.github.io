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
y = \sum_{m \in \operatorname{TopK}(r(x))}
g_m(x) E_m(x)
$$

Here $r(x)$ is the router, $E_m$ is expert $m$, and $g_m(x)$ is the routing weight.

## Key Ideas

- A router selects one or more experts for each token, example, node, or region.
- Sparse MoE increases parameter count while activating only a subset of parameters per input.
- Experts are often feed-forward blocks, but the pattern can apply to other modules.
- Load-balancing losses or routing constraints keep a few experts from receiving all traffic.
- Routing can create specialization, but specialization should be checked rather than assumed.

## Practical Checks

- Check routing granularity: token-level, sequence-level, graph-level, modality-level, or task-level.
- Track top-k choice, capacity limits, dropped tokens, and auxiliary routing losses.
- Compare parameter count, activated parameter count, memory traffic, and inference latency separately.
- For agent or tool systems, distinguish MoE routing from explicit [[agents/tool-use|tool use]].

## Related

- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/mlp|MLP]]
- [[agents/index|Agents]]
- [[concepts/learning/index|Learning methods]]
