---
title: Scaling Claim Contract
tags:
  - systems
  - evaluation
  - architectures
---

# Scaling Claim Contract

A scaling claim contract records what changes when a paper says a method scales better. Scaling can mean better quality at larger data, lower memory at longer sequence length, faster inference, or better quality per unit compute. These are different claims.

The compact form is:

$$
\text{scaling claim}
=
(\text{quality},\ \text{data},\ \text{model},\ \text{compute},\ \text{runtime})
$$

Without all five parts, a scaling result is easy to overread.

## Claim Types

| Claim | Must Specify | Common Trap |
| --- | --- | --- |
| Model-size scaling | parameter count, activated parameter count, architecture family | comparing a larger model to a smaller baseline |
| Data scaling | dataset size, filtering, deduplication, epoch/token budget | more data also changes data quality |
| Compute scaling | FLOPs, accelerator type, training time, precision, batch size | wall time and FLOPs are not interchangeable |
| Length scaling | sequence length, graph size, atom count, image resolution, context length | benchmark size is smaller than deployment size |
| Memory scaling | parameter memory, activation memory, KV cache, optimizer state | peak memory hides batch and implementation details |
| Inference scaling | latency, throughput, batch policy, request shape | speed is reported without holding quality fixed |
| Quality-per-compute | metric improvement per cost unit | metric changes while compute also changes |

## Budget Variables

For training, record:

$$
B_{\mathrm{train}}
=
(N_{\mathrm{data}},\ N_{\mathrm{params}},\ T_{\mathrm{steps}},\ B_{\mathrm{batch}},\ P,\ H)
$$

where $N_{\mathrm{data}}$ is dataset size, $N_{\mathrm{params}}$ is parameter count, $T_{\mathrm{steps}}$ is optimizer steps or consumed samples/tokens, $B_{\mathrm{batch}}$ is effective batch size, $P$ is precision, and $H$ is hardware.

For inference, record:

$$
B_{\mathrm{infer}}
=
(L_{\mathrm{in}},\ L_{\mathrm{out}},\ B_{\mathrm{serve}},\ H,\ P,\ \text{cache policy})
$$

where $L_{\mathrm{in}}$ and $L_{\mathrm{out}}$ are input and output size, $B_{\mathrm{serve}}$ is serving batch or concurrency, and cache policy includes KV cache, feature cache, graph cache, or conformer cache when relevant.

## Fair Comparison

A fair scaling comparison should state which variables are controlled:

| Controlled | Free to Vary | Supported Claim |
| --- | --- | --- |
| Same compute, same data | architecture or objective | more efficient method under fixed budget |
| Same model family, same compute | data size | data scaling behavior |
| Same data, same architecture | model size | parameter scaling behavior |
| Same quality target | latency, memory, cost | efficiency at fixed utility |
| Same latency target | quality, memory, cost | quality under serving constraint |

If multiple variables change at once, the result is a bundled system result, not clean evidence for one architecture, objective, or dataset change.

## Scaling Curve

When several budgets are tested, read the curve rather than only the best point:

$$
M(b)
=
\text{metric at budget } b
$$

The important questions are:

$$
\Delta M
=
M(b_2)-M(b_1),
\qquad
\frac{\Delta M}{\Delta b}
=
\text{marginal gain per budget}
$$

For many papers, the practical claim is not "best score" but "useful marginal gain remains under the target budget."

## Molecular Modeling Notes

For molecular modeling and structure-based work, scaling variables can be domain-specific:

| Variable | Meaning |
| --- | --- |
| Atom count | ligand, protein, pocket, or complex size |
| Conformer count | number of generated or evaluated geometries |
| Candidate count | virtual screening library size |
| Target count | number of protein targets or assay contexts |
| Template/retrieval count | number of structures, homologs, or retrieved contexts |
| Docking budget | poses per ligand, exhaustiveness, minimization, rescoring |

Do not compare a learned model to docking or physics baselines without stating candidate count, pose count, conformer policy, and postprocessing budget.

## Paper Reading Checks

- What is claimed to scale: quality, length, memory, latency, throughput, cost, or data?
- Which budget variables were held fixed?
- Are parameter count and activated parameter count separated?
- Is training compute separated from inference compute?
- Are FLOPs, wall time, hardware, precision, and implementation reported?
- Is the baseline run with the same data, preprocessing, and model-selection budget?
- Does the metric improve at the same serving constraint, or only with more compute?
- Are failed runs, out-of-memory cases, truncation, or timeout included?
- For molecular modeling, are conformer, docking, candidate, and postprocessing budgets included?

## Related

- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/architectures/architecture-selection|Architecture selection]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[concepts/machine-learning/training-step-accounting|Training step accounting]]
- [[concepts/machine-learning/model-state-contract|Model state contract]]
- [[concepts/evaluation/benchmark-claim-contract|Benchmark claim contract]]
- [[papers/workflows/claim-routing|Claim routing]]
