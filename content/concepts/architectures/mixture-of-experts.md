---
title: Mixture of Experts
tags:
  - architectures
  - mixture-of-experts
  - routing
---

# Mixture of Experts

Mixture-of-experts models route each input through a subset of expert modules. The pattern separates model capacity from the amount of computation used per token or example.

## Uses

- Sparse scaling for large [[concepts/architectures/transformer|Transformer]] models.
- Specialist modules selected by learned routing.
- Agent backbones where capacity and inference cost matter.

## Related

- [[concepts/architectures/transformer|Transformer]]
- [[agents/index|Agents]]
- [[concepts/learning/index|Learning methods]]
