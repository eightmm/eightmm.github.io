---
title: GLaM
aliases:
  - papers/glam
  - papers/generalist-language-model
  - papers/efficient-scaling-language-models-moe
tags:
  - papers
  - architectures
  - mixture-of-experts
  - language-models
  - conditional-compute
---

# GLaM

> GLaM shows how sparse Mixture-of-Experts routing can be used as a large autoregressive language-model scaling route.

## Metadata

| Field | Value |
| --- | --- |
| Paper | GLaM: Efficient Scaling of Language Models with Mixture-of-Experts |
| Authors | Nan Du, Yanping Huang, Andrew M. Dai, Simon Tong, Dmitry Lepikhin, Yuanzhong Xu, Maxim Krikun, Yanqi Zhou, Adams Wei Yu, Orhan Firat, Barret Zoph, Liam Fedus, Maarten Bosma, Zongwei Zhou, Tao Wang, Yu Emma Wang, Kellie Webster, Marie Pellat, Kevin Robinson, Kathleen Meier-Hellstern, Toju Duke, Lucas Dixon, Kun Zhang, Quoc V. Le, Yonghui Wu, Zhifeng Chen, Claire Cui |
| Year | 2021 preprint; 2022 conference |
| Venue | ICML 2022 |
| arXiv | [2112.06905](https://arxiv.org/abs/2112.06905) |
| PMLR | [PMLR 162:5547-5569](https://proceedings.mlr.press/v162/du22c.html) |
| Status | full note started |

## Question

Dense autoregressive language models scale by increasing parameters, data, and training compute. This creates a simple but expensive relationship:

$$
\text{larger dense model}
\Rightarrow
\text{more active compute per token}.
$$

GLaM asks:

$$
\text{Can a language model increase total capacity while activating only a sparse subset per token?}
$$

The answer is a sparse [[concepts/architectures/mixture-of-experts|Mixture of Experts]] decoder-only language model.

## Main Claim

GLaM uses sparsely activated MoE layers inside an autoregressive language model so that total parameter capacity grows faster than per-token computation.

The durable architecture claim is:

$$
\text{decoder-only Transformer}
+
\text{interleaved sparse MoE FFN layers}
+
\text{top-2 expert routing}
\Rightarrow
\text{large-capacity LM with lower active compute than dense scaling}.
$$

This makes GLaM a bridge between [[papers/architectures/gpt-3|GPT-3]]-style dense scaling and later sparse expert LLMs.

## Architecture Contract

| Component | Contract |
| --- | --- |
| Base model | autoregressive decoder-only language model |
| Token state | $x_t \in \mathbb{R}^{d}$ |
| Dense path | self-attention and non-MoE Transformer layers |
| Sparse path | selected MoE feed-forward experts in some layers |
| Router | token-level gate over experts |
| Expert selection | sparse top-2 expert activation |
| Output | next-token distribution |
| Main comparison | dense models under compute, quality, and energy constraints |

The training objective remains the causal language-model objective:

$$
\mathcal{L}
=
-\sum_{t=1}^{T}
\log p_\theta(x_t \mid x_{<t}).
$$

The architecture change is in the conditional feed-forward path, not the output objective.

## Sparse MoE Layer

Let a token hidden state be:

$$
x_t \in \mathbb{R}^{d}.
$$

A dense Transformer FFN computes:

$$
\operatorname{FFN}(x_t)
=
W_2 \sigma(W_1 x_t + b_1)+b_2.
$$

GLaM replaces some dense FFN layers with an expert pool:

$$
E_1, E_2, \dots, E_M.
$$

The router computes expert scores:

$$
r_t
=
\operatorname{softmax}(W_r x_t),
\qquad
r_t \in \mathbb{R}^{M}.
$$

For top-2 routing:

$$
S_t
=
\operatorname{TopK}(r_t,2).
$$

The MoE output is:

$$
\operatorname{MoE}(x_t)
=
\sum_{i\in S_t}
\alpha_{t,i}E_i(x_t),
$$

where $\alpha_{t,i}$ is the normalized gate weight for selected expert $i$.

The key scaling relation is:

$$
|\theta|_{\mathrm{total}}
\gg
|\theta|_{\mathrm{active}}(x_t).
$$

## Interleaving Dense and Sparse Layers

GLaM is not "all experts everywhere." The important pattern is a dense Transformer backbone with MoE layers inserted in place of selected feed-forward blocks.

At a high level:

$$
h^{\ell+1}
=
\operatorname{Attention}(h^\ell)
+
\operatorname{MoE\text{-}FFN}(h^\ell)
$$

for sparse layers, while dense layers use:

$$
h^{\ell+1}
=
\operatorname{Attention}(h^\ell)
+
\operatorname{DenseFFN}(h^\ell).
$$

This matters because the sparse route increases model capacity while retaining the familiar decoder-only Transformer interface.

## Capacity, Active Compute, and Reporting

GLaM is easy to misread if total parameters and active computation are not separated.

| Quantity | Meaning |
| --- | --- |
| total parameters | all expert parameters plus shared Transformer parameters |
| active parameters | parameters used by selected experts for one token |
| FLOPs per token | computation actually used during forward pass |
| routing overhead | gate, dispatch, expert batching, and combination cost |
| energy/training cost | system-level cost, not just architecture parameter count |

The right reading contract is:

$$
\text{parameter count}
\neq
\text{active compute}
\neq
\text{wall-clock cost}.
$$

## Relation to Prior MoE Papers

| Paper | What It Establishes |
| --- | --- |
| [Sparsely-Gated MoE](/papers/architectures/sparsely-gated-moe) | sparse expert routing and load balancing as a reusable layer |
| [Switch Transformer](/papers/architectures/switch-transformer) | top-1 sparse Transformer FFN routing for simpler scaling |
| GLaM | large autoregressive LM scaling with sparse top-2 MoE layers |

The lineage is:

$$
\text{conditional compute layer}
\rightarrow
\text{sparse Transformer FFN}
\rightarrow
\text{sparse expert language model}.
$$

## Relation to GPT-3 and LLaMA

| Axis | GPT-3 | GLaM | LLaMA |
| --- | --- | --- | --- |
| main route | dense decoder-only scaling | sparse expert decoder-only scaling | efficient open dense decoder-only recipe |
| FFN path | dense FFN | MoE FFN in selected layers | dense SwiGLU-style FFN |
| capacity | all active per token | many experts, sparse activation | dense active parameters |
| reading focus | in-context learning at scale | compute-efficient sparse scaling | modern open-weight implementation recipe |

GLaM should not be read as replacing dense LLMs. It shows that sparse capacity is a serious architecture route when training and inference cost matter.

## Evidence

| Evidence Type | What It Supports |
| --- | --- |
| zero-shot and one-shot task suite | sparse MoE language models can compete with large dense baselines |
| compute and energy reporting | sparse activation can reduce training and inference cost relative to dense scaling claims |
| model-size family | MoE scaling behavior should be evaluated across multiple capacity points |
| dense baseline comparisons | quality claims depend on compute-matched or cost-aware comparisons |

## Evaluation Risks

- Total parameter count can make the model look much larger than the active computation per token.
- Energy and FLOP comparisons depend on hardware, implementation, utilization, and accounting boundaries.
- Sparse MoE routing can create load imbalance and serving complexity not visible in simple FLOP counts.
- Data quality, data mixture, and benchmark choice may explain part of the gain.
- Expert specialization should be measured; it should not be assumed from the presence of experts.

## Why It Belongs in Architecture Papers

GLaM belongs here because it changes the scaling contract of a language-model architecture:

$$
\text{dense LM scaling}
\rightarrow
\text{sparse expert LM scaling}.
$$

It is also a useful bridge for reading later MoE LLMs, where the important questions are:

- how many experts exist per MoE layer;
- how many experts are active per token;
- which layers are sparse;
- how the router is trained and balanced;
- whether reported compute includes routing, communication, and serving overhead.

## Concepts

- [[concepts/architectures/mixture-of-experts|Mixture of Experts]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
- [[concepts/systems/latency-throughput|Latency and throughput]]

## Related

- [[papers/architectures/sparsely-gated-moe|Sparsely-Gated MoE]]
- [[papers/architectures/switch-transformer|Switch Transformer]]
- [[papers/architectures/gpt-3|GPT-3]]
- [[papers/architectures/llama|LLaMA]]
