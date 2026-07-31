---
title: Architecture Search
tags:
  - concepts
  - architectures
  - neural-architecture-search
---

# Architecture Search

Architecture search treats a model description as an object to optimize over a constrained search space. It is different from ordinary hyperparameter tuning because the decisions can change the computation graph, parameter sharing, inductive bias, and input/output contract.

Let $m\in\mathcal{M}$ be a candidate architecture and $w_m^*$ its trained parameters. A basic search objective is:

$$
m^*
=
\arg\max_{m\in\mathcal{M}}
R\left(m,w_m^*;D_{\mathrm{valid}}\right).
$$

## Search components

| Component | Question |
| --- | --- |
| Search space | Which operations, connections, widths, depths, and constraints are allowed? |
| Search strategy | How are candidate architectures proposed? |
| Fidelity | How much data, time, and training does each candidate receive? |
| Reward | Which quality, latency, memory, or energy signal is optimized? |
| Selection | How are candidates retrained and compared after search? |

Common search strategies include random search, evolutionary methods, reinforcement-learning controllers, Bayesian optimization, and differentiable relaxations. The method should not be evaluated independently from the space and budget.

## Cost boundary

Search cost and deployment cost are different quantities:

$$
\text{total search cost}
\approx
\text{number of candidates}
\times
\text{candidate training cost}.
$$

A model can be cheap to serve but expensive to discover. Conversely, weight sharing or low-fidelity proxies can reduce search cost while introducing ranking bias.

## Evaluation checklist

- Compare with random search in the same space.
- Match candidate training budgets and data conditions.
- Report the complete search space and constraints.
- Retrain the selected architecture from independent seeds.
- Measure target-hardware latency and memory rather than relying only on FLOPs.
- Separate validation data used during search from final test evidence.

## Related

- [[papers/architectures/neural-architecture-search-with-reinforcement-learning|Neural Architecture Search with Reinforcement Learning]]
- [[concepts/architectures/architecture-selection|Architecture selection]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[papers/architectures/efficientnet|EfficientNet]]
