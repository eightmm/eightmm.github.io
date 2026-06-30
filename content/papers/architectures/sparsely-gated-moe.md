---
title: Sparsely-Gated Mixture-of-Experts
aliases:
  - papers/sparsely-gated-moe
  - papers/outrageously-large-neural-networks
  - papers/mixture-of-experts-layer
tags:
  - papers
  - architectures
  - mixture-of-experts
  - routing
  - conditional-compute
---

# Sparsely-Gated Mixture-of-Experts

> The paper made conditional computation practical by routing each example or token through a sparse subset of expert networks.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer |
| Authors | Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean |
| Year | 2017 |
| Venue | ICLR 2017 |
| arXiv | [1701.06538](https://arxiv.org/abs/1701.06538) |
| OpenReview | [B1ckMDqlg](https://openreview.net/forum?id=B1ckMDqlg) |
| Status | full note started |

## Question

Dense neural networks tie capacity and computation together. If a layer has many parameters, every input usually pays for most of those parameters.

The paper asks:

$$
\text{Can model capacity grow far faster than per-example computation?}
$$

The proposed answer is conditional computation: keep many expert modules available, but activate only a small subset for each input.

## Main Claim

A sparsely-gated [[concepts/architectures/mixture-of-experts|Mixture of Experts]] layer can increase total parameter capacity while keeping active computation relatively small.

The durable architecture claim is:

$$
\text{large expert pool}
+
\text{sparse trainable router}
+
\text{load-balancing constraints}
\Rightarrow
\text{high capacity under conditional compute}.
$$

This paper is the canonical predecessor to later sparse Transformer MoE models such as [[papers/architectures/switch-transformer|Switch Transformer]].

## Architecture Contract

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| input representation | $x \in \mathbb{R}^{d}$ | router input | decide which experts should run |
| gating network | $x$ | sparse gate vector $g(x)$ | choose and weight experts |
| expert networks | $x$ routed to selected experts | expert outputs | provide large conditional capacity |
| dispatcher | batch of inputs | per-expert mini-batches | group inputs by selected experts |
| combiner | expert outputs and gates | layer output | merge selected expert outputs |
| load losses | router statistics | auxiliary loss | prevent expert collapse |

The key contract is:

$$
\text{many parameters exist, few parameters are active per input}.
$$

## MoE Layer

Let there be $M$ expert networks:

$$
E_1, E_2, \dots, E_M.
$$

For an input $x$, a gating network produces expert scores:

$$
h(x) = W_g x.
$$

The dense mixture form would be:

$$
y = \sum_{i=1}^{M} g_i(x) E_i(x).
$$

That is expensive when $M$ is large. The sparsely-gated version selects only the top-$k$ experts:

$$
S(x)=\operatorname{TopK}(h(x), k).
$$

The output is:

$$
y
=
\sum_{i \in S(x)}
g_i(x)E_i(x),
$$

where $g_i(x)$ is nonzero only for selected experts.

## Noisy Top-k Routing

Pure top-$k$ routing can become brittle: small score changes can switch experts, and the router may collapse to a small subset of experts.

The paper uses noisy gating:

$$
H_i(x)
=
(xW_g)_i
+
\epsilon_i \operatorname{Softplus}((xW_{\text{noise}})_i),
$$

where $\epsilon_i$ is noise and $W_{\text{noise}}$ controls noise scale.

The sparse gate is then computed over the selected experts:

$$
G_i(x)
=
\frac{\exp(H_i(x))}
{\sum_{j\in S(x)}\exp(H_j(x))}
\quad
\text{if } i\in S(x),
$$

and:

$$
G_i(x)=0
\quad
\text{if } i\notin S(x).
$$

The output becomes:

$$
y
=
\sum_{i=1}^{M}
G_i(x)E_i(x).
$$

## Why Load Balancing Is Part of the Architecture

Sparse MoE is not just a mathematical layer. It creates a routing system. If most inputs choose the same expert, capacity is wasted and computation becomes imbalanced.

For a batch $\mathcal{B}$, define expert load:

$$
L_i
=
\sum_{x\in \mathcal{B}}
\mathbf{1}[i\in S(x)].
$$

Balanced routing wants each $L_i$ to be close to:

$$
\frac{|\mathcal{B}|k}{M}.
$$

The paper adds auxiliary losses to encourage:

$$
\text{importance balance}
\quad\text{and}\quad
\text{load balance}.
$$

This became a lasting design principle for sparse MoE papers: routing quality and systems balance are part of the model, not an implementation detail.

## Conditional Compute Scaling

Dense scaling roughly follows:

$$
\text{more parameters}
\Rightarrow
\text{more active compute}.
$$

Sparse MoE tries to change the scaling relationship:

$$
|\theta|_{\mathrm{total}}
\gg
|\theta|_{\mathrm{active}}(x).
$$

The important distinction is:

| Quantity | Meaning |
| --- | --- |
| total parameters | all expert parameters plus shared model parameters |
| active parameters | parameters used by selected experts for a given input |
| routing cost | gate computation and dispatch/combination overhead |
| communication cost | moving token/expert batches across devices |

A paper that reports only total parameters can make sparse models look larger than the computation each token uses. A paper that reports only FLOPs can hide routing and communication costs.

## Relation to Switch Transformer

| Axis | Sparsely-Gated MoE | Switch Transformer |
| --- | --- | --- |
| base model in paper | LSTM/convolutional sequence models | Transformer |
| router | noisy top-$k$ gate | simplified top-1 router |
| expert type | feed-forward expert sub-networks | Transformer FFN experts |
| central problem | make conditional compute practical | simplify sparse Transformer scaling |
| lasting concept | sparse expert routing | top-1 expert routing at Transformer scale |

The lineage is:

$$
\text{Sparsely-Gated MoE}
\rightarrow
\text{Transformer MoE}
\rightarrow
\text{Switch-style top-1 routing}.
$$

## Evidence

| Evidence Type | What It Supports |
| --- | --- |
| large expert-count experiments | sparse routing can support much larger parameter counts than dense active compute |
| language modeling and machine translation results | conditional capacity can improve sequence modeling tasks |
| efficiency analysis | active compute and total capacity can be partially decoupled |
| routing/load analysis | auxiliary balancing is needed for practical expert utilization |

## Evaluation Risks

- Total parameter count does not equal active compute.
- Sparse MoE may improve quality because of extra capacity, not because experts are interpretable.
- Load balancing losses can change task loss and specialization behavior.
- Wall-clock speed depends on dispatch, batching, device placement, and communication.
- Expert specialization should be measured from routing behavior, not inferred from expert IDs.

## Why It Belongs in Architecture Papers

The paper defines a reusable layer contract:

$$
x
\rightarrow
\operatorname{router}(x)
\rightarrow
\{E_i(x)\}_{i\in S(x)}
\rightarrow
y.
$$

That contract is now central to sparse LLMs, multimodal routing, large-capacity Transformers, and systems-aware model design.

## Concepts

- [[concepts/architectures/mixture-of-experts|Mixture of Experts]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
- [[concepts/systems/latency-throughput|Latency and throughput]]

## Related

- [[papers/architectures/switch-transformer|Switch Transformer]]
- [[papers/architectures/glu-variants-improve-transformer|GLU Variants Improve Transformer]]
- [[papers/architectures/t5|T5]]
- [[papers/architectures/llama|LLaMA]]
