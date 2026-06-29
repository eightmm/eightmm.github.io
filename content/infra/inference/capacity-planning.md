---
title: Inference Capacity Planning
unlisted: true
tags:
  - infra
  - inference
---

# Inference Capacity Planning

Capacity planning sits between model behavior and hardware limits. Start from [[concepts/systems/inference-capacity-planning|Inference capacity planning]] for the reusable systems concept; infra notes provide the resource mental model.

| Question | Go To |
| --- | --- |
| How many requests fit under latency and memory limits? | [Inference capacity planning](/concepts/systems/inference-capacity-planning) |
| What is the serving path? | [Inference serving](/concepts/systems/inference-serving) |
| What hardware limit matters? | [Hardware](/infra/hardware/), [GPU](/infra/gpu/) |
| How should results be recorded? | [Reproducible run record](/infra/reproducibility/run-record) |

Public capacity notes should use generic workloads and formulas. Do not publish real traffic, private utilization, endpoint names, account names, dashboards, or unpublished benchmark results.
