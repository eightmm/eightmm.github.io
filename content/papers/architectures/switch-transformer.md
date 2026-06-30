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

> The paper makes sparse Mixture-of-Experts scaling simpler by routing each token to one expert.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity |
| Authors | William Fedus, Barret Zoph, Noam Shazeer |
| Year | 2021 preprint; 2022 journal |
| Venue | JMLR 2022 |
| arXiv | [2101.03961](https://arxiv.org/abs/2101.03961) |
| JMLR | [23(120):1-39](https://jmlr.org/papers/v23/21-0998.html) |
| Status | full note started |

## One-Line Takeaway

Switch Transformer replaces dense feed-forward layers with sparse expert layers and uses top-1 routing so model capacity can grow faster than per-token compute.

## Question

Dense Transformers use the same parameters for every token. If a model has hidden states

$$
X = [x_1,\dots,x_T], \qquad x_t \in \mathbb{R}^{d},
$$

then a standard Transformer block applies the same feed-forward network to every token:

$$
y_t = \operatorname{FFN}(x_t).
$$

Increasing model size usually increases the computation each token must pay. If width grows from $d$ to $D$, or if more layers are added, every token uses more FLOPs.

Mixture-of-Experts changes this contract:

$$
\text{total parameters} \gg \text{parameters used per token}.
$$

The paper asks:

> Can sparse expert routing be simplified enough to make very large parameter-count Transformers practical to train?

## Main Claim

Top-1 expert routing is sufficient for effective sparse Transformer scaling. Instead of mixing multiple expert outputs, each token is sent to one selected expert.

For $N$ experts:

$$
E_1, E_2, \dots, E_N,
$$

the router computes a probability distribution:

$$
p_t = \operatorname{softmax}(W_r x_t),
$$

where:

$$
p_t \in \mathbb{R}^{N}.
$$

The selected expert is:

$$
e_t = \arg\max_i p_{t,i}.
$$

The Switch layer output is approximately:

$$
y_t = p_{t,e_t} E_{e_t}(x_t).
$$

The key simplification is:

$$
\text{one token} \rightarrow \text{one expert}.
$$

Earlier MoE variants often used top-2 routing:

$$
y_t = \sum_{i \in \operatorname{TopK}(p_t,2)} p_{t,i} E_i(x_t).
$$

Switch uses:

$$
y_t = p_{t,e_t}E_{e_t}(x_t).
$$

This reduces routing complexity, expert communication, and expert combination cost.

## Architecture Contract

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Token hidden state | $x_t \in \mathbb{R}^d$ | router logits | choose expert |
| Router | token state | expert index and gate value | conditional path |
| Expert FFN | assigned tokens | transformed token states | sparse capacity |
| Capacity rule | token assignments | accepted/dropped tokens | bound per-expert load |
| Load-balancing loss | router statistics | auxiliary scalar loss | prevent expert collapse |

The Switch layer replaces the dense FFN sublayer in a Transformer block:

$$
\operatorname{TransformerBlock}
=
\operatorname{Attention}
+
\operatorname{SwitchFFN}.
$$

Attention remains dense. Sparsity is applied mainly to the feed-forward block.

## Dense FFN Baseline

A standard Transformer feed-forward network is:

$$
\operatorname{FFN}(x)
=
W_2 \sigma(W_1 x + b_1) + b_2.
$$

For hidden width $d$ and intermediate width $d_{\text{ff}}$, parameter count is roughly:

$$
2 d d_{\text{ff}}.
$$

Every token uses this same computation:

$$
O(d d_{\text{ff}}).
$$

If the dense FFN is made wider, both parameter count and per-token compute increase.

## Expert FFN Layer

Switch Transformer creates $N$ expert FFNs:

$$
\operatorname{Expert}_i(x)
=
W_{2,i}\sigma(W_{1,i}x+b_{1,i})+b_{2,i}.
$$

Total expert parameters become:

$$
N \cdot 2 d d_{\text{ff}}.
$$

But each token only evaluates one expert:

$$
\operatorname{SwitchFFN}(x_t)
=
p_{t,e_t}\operatorname{Expert}_{e_t}(x_t).
$$

So the expert parameter pool grows with $N$, while per-token expert compute remains close to one FFN:

$$
\text{expert compute per token} \approx O(d d_{\text{ff}}).
$$

This is conditional computation.

## Router

The router is usually a linear classifier over experts:

$$
z_t = W_r x_t,
$$

$$
p_t = \operatorname{softmax}(z_t).
$$

The top expert is:

$$
e_t = \arg\max_i p_{t,i}.
$$

The gate value is:

$$
g_t = p_{t,e_t}.
$$

The output is:

$$
y_t = g_t E_{e_t}(x_t).
$$

This means routing is discrete in expert choice but still uses a continuous gate multiplier.

## Capacity Factor

If every token chooses the same expert, sparse routing breaks. One expert receives too many tokens, other experts are idle, and distributed training becomes imbalanced.

For a batch with $T$ tokens and $N$ experts, the ideal tokens per expert are:

$$
\frac{T}{N}.
$$

Switch defines an expert capacity:

$$
C
=
\left\lceil
\frac{T}{N} \cdot \alpha
\right\rceil,
$$

where $\alpha$ is the capacity factor.

If expert $i$ receives more than $C$ tokens, extra tokens are usually dropped or bypassed depending on implementation. This introduces a real architectural tradeoff:

| Capacity Factor | Effect |
| --- | --- |
| low $\alpha$ | better memory/compute bound, more token drops |
| high $\alpha$ | fewer drops, worse load and memory overhead |

Capacity factor is not a small engineering detail. It changes model quality, throughput, and training stability.

## Load-Balancing Loss

The router needs pressure to use experts evenly. Let:

$$
f_i
$$

be the fraction of tokens routed to expert $i$, and let:

$$
P_i
$$

be the average router probability assigned to expert $i$.

A common Switch-style auxiliary objective is:

$$
L_{\text{aux}}
=
\lambda N \sum_{i=1}^{N} f_i P_i.
$$

The goal is not to make every token equally likely for every expert. The goal is to discourage collapse where one expert dominates the batch.

The full training loss becomes:

$$
L = L_{\text{task}} + L_{\text{aux}}.
$$

For language modeling:

$$
L_{\text{task}}
=
-\sum_t \log p_\theta(x_t \mid x_{<t}).
$$

## Why Top-1 Routing Matters

Top-2 routing gives each token two expert paths:

$$
y_t
=
g_{t,a}E_a(x_t)
+
g_{t,b}E_b(x_t).
$$

This can improve routing robustness, but it doubles expert computation and increases communication.

Top-1 routing uses:

$$
y_t = g_t E_{e_t}(x_t).
$$

This has several practical consequences:

- fewer expert calls per token;
- simpler combine/scatter logic;
- lower communication volume;
- easier scaling to many experts;
- more sensitivity to routing mistakes.

The paper's bet is that the simplicity is worth the reduced redundancy.

## Where Sparsity Is and Is Not

Switch Transformer is sparse in the FFN path. It is not sparse everywhere.

| Block | Dense or Sparse | Comment |
| --- | --- | --- |
| token embedding | dense | every token embedded |
| self-attention | dense | all tokens attend according to standard attention pattern |
| expert FFN | sparse | each token uses one expert |
| residual stream | dense | all tokens continue through shared layers |
| output head | dense | task-dependent |

This is why "trillion parameter model" does not mean every token touches a trillion parameters.

## Scaling Interpretation

Dense scaling:

$$
\text{more parameters} \Rightarrow \text{more compute per token}.
$$

Sparse MoE scaling:

$$
\text{more parameters} \not\Rightarrow \text{same-rate increase in compute per token}.
$$

The useful metric is not just parameter count:

$$
\text{total parameters}
$$

but also:

$$
\text{active parameters per token},
$$

$$
\text{tokens/sec},
$$

$$
\text{communication overhead},
$$

$$
\text{quality at fixed compute}.
$$

This is why Switch belongs in both architecture and systems discussions.

## Communication Pattern

In distributed training, experts may be sharded across devices. Routing then requires token dispatch:

$$
\text{tokens on device A}
\rightarrow
\text{experts on devices B,C,D,\dots}
\rightarrow
\text{return processed tokens}.
$$

This creates all-to-all communication. A sparse expert layer can be compute-efficient but communication-heavy.

The real throughput depends on:

- batch size;
- sequence length;
- number of experts;
- expert placement;
- network bandwidth;
- capacity factor;
- token dropping rate;
- precision format;
- implementation kernel quality.

This is the main reason MoE is an architecture-systems paper, not just a modeling paper.

## Evidence Reading

The paper's evidence is centered on language-model pretraining and transfer. The core result is that sparse expert capacity can improve scaling efficiency compared with dense T5-like baselines under the evaluated setup.

| Claim | Evidence Type | What It Supports | Caveat |
| --- | --- | --- | --- |
| Top-1 routing can train well | comparison to dense and sparse baselines | one-expert routing is viable | tied to LM setup and routing details |
| Sparse models improve compute efficiency | pretraining loss and transfer results | more parameters can help at similar active compute | not equivalent to universal speedup |
| Very large parameter counts are trainable | trillion-parameter-scale experiments | conditional compute enables huge models | systems stack matters |
| Load balancing is necessary | auxiliary loss and capacity design | router collapse is a real failure mode | balancing may fight specialization |

## Benchmark Card

| Field | Value |
| --- | --- |
| Main task family | language-model pretraining and transfer |
| Architecture family | sparse Mixture-of-Experts Transformer |
| Replaced component | Transformer FFN sublayer |
| Routing | top-1 expert choice |
| Main comparison | dense T5-style Transformer baselines |
| Key metrics | pretraining loss, downstream task quality, speed, parameter count |
| Core hyperparameters | number of experts $N$, capacity factor $\alpha$, auxiliary loss weight $\lambda$ |
| Not directly tested | universal MoE behavior in all modalities or small-data regimes |

## Comparison to Dense Transformer

| Property | Dense Transformer | Switch Transformer |
| --- | --- | --- |
| FFN path | shared FFN for all tokens | routed expert FFN |
| Parameters used per token | most layer parameters | one expert plus shared blocks |
| Total capacity | tied to dense width/depth | grows with expert count |
| Routing | none | learned token-to-expert router |
| Main systems issue | dense compute | dispatch, all-to-all, imbalance |
| Failure mode | undercapacity or overcompute | expert collapse, token drops, routing noise |

Dense Transformers are simpler and often better for small/medium regimes. Switch becomes attractive when the system can support sparse routing and the workload benefits from large capacity.

## Comparison to Earlier MoE

Earlier sparse MoE designs commonly used top-k routing, often top-2. Switch simplifies this to top-1.

| Design | Routing | Benefit | Cost |
| --- | --- | --- | --- |
| Top-2 MoE | two experts per token | redundancy, smoother routing | more compute and communication |
| Switch | one expert per token | simpler, faster, easier scaling | less routing redundancy |

The paper's contribution is not inventing MoE. It is showing that a simplified version can scale effectively.

## Expert Specialization

A tempting interpretation is:

$$
\text{expert} = \text{human-readable skill}.
$$

That is not guaranteed. Experts may specialize by:

- token type;
- language;
- syntax;
- frequency;
- domain;
- position;
- optimization artifact;
- hardware load pattern.

Router analysis is needed before making semantic claims about experts.

## Failure Modes

### Expert Collapse

If many tokens choose one expert:

$$
f_i \gg \frac{1}{N}
$$

for some expert $i$, the model loses parallelism and effective capacity.

### Token Dropping

If an expert exceeds capacity:

$$
\#\{t : e_t=i\} > C,
$$

some tokens cannot be processed by that expert. This can degrade quality, especially if dropped tokens are not random.

### Router Instability

The router is trained jointly with the model. Small changes in hidden states can change expert assignment:

$$
\arg\max_i p_{t,i}
$$

can flip discontinuously. This makes optimization more delicate than a dense FFN.

### Communication Bottleneck

If all-to-all communication dominates, sparse compute does not translate into wall-clock speedup.

## Implementation Notes

### Batch and Sequence Shape

For hidden states:

$$
X \in \mathbb{R}^{B \times T \times d},
$$

tokens are often flattened:

$$
X_{\text{flat}} \in \mathbb{R}^{BT \times d}.
$$

The router maps:

$$
X_{\text{flat}} \rightarrow P \in \mathbb{R}^{BT \times N}.
$$

Assignments:

$$
e_t = \arg\max_i P_{t,i}.
$$

Then tokens are grouped by expert, processed, and scattered back to original order.

### Precision

Sparse routing can be numerically sensitive. Router logits and probabilities may need more stable precision than the expert FFN path, depending on implementation.

### Metrics to Log

For an MoE run, track:

- expert load histogram;
- token drop rate;
- router entropy;
- auxiliary loss;
- per-expert throughput;
- all-to-all communication time;
- quality at fixed active FLOPs;
- quality at fixed wall-clock time.

Without these metrics, parameter count claims are hard to interpret.

## Molecular and Structural Modeling Reading

Switch-style MoE can be relevant outside language, but it should be treated carefully.

Possible uses:

- route tokens by modality: protein, ligand, text, assay metadata;
- route residues/atoms by local environment type;
- allocate experts to different sequence families or structural regimes;
- build large conditional-capacity models where only part of the model activates per example;
- specialize experts for tasks such as scoring, generation, and refinement.

Risks:

- small scientific datasets may not support many experts;
- routing can learn dataset artifacts;
- expert specialization can amplify data imbalance;
- distributed MoE complexity may outweigh modeling benefit;
- if the task needs geometric equivariance, MoE does not provide it by itself.

For structure-based modeling, MoE is usually a capacity/scaling mechanism, not the core inductive bias. It should complement graph, geometric, or sequence architecture choices rather than replace them.

## Common Misreadings

### "A trillion-parameter sparse model uses a trillion parameters per token."

No. The model may have a huge total parameter count, but each token activates only a subset.

### "MoE is just an ensemble."

It is not a standard ensemble because experts are inside one model, trained jointly, and selected per token by a learned router.

### "Top-1 routing is obviously worse than top-2."

Top-2 has redundancy, but top-1 can win on simplicity, speed, and scaling. The tradeoff is empirical and systems-dependent.

### "Load balancing means all experts learn the same thing."

Load balancing encourages even usage. It does not force identical functions. In fact, useful MoE wants both balanced usage and meaningful specialization.

### "Sparse compute removes systems bottlenecks."

Sparse compute changes the bottlenecks. Communication, routing, and capacity management become central.

## Later-Paper Checklist

When reading later MoE, routing, sparse scaling, or expert-model papers, ask:

- Is routing top-1, top-2, top-k, hash-based, task-based, or learned?
- What is the active parameter count per token?
- What is the total parameter count?
- Is quality compared at fixed FLOPs, fixed wall-clock, or fixed parameter count?
- What is the token drop rate?
- How is expert load balanced?
- Does the paper report router entropy or load histograms?
- Is the speedup measured end-to-end or only theoretical?
- Does routing specialize semantically or only balance systems load?
- What happens in small-data or domain-shift settings?

## Why It Matters

Switch Transformer is a key architecture paper because it made sparse expert scaling feel operationally simple:

$$
\text{replace FFN}
\rightarrow
\text{route each token to one expert}
\rightarrow
\text{scale parameter capacity}.
$$

It is also a useful reminder that modern architecture is often inseparable from systems. The architectural idea only works if routing, dispatch, memory, and communication are handled well.

## Limitations

- Evidence is centered on language-model pretraining and transfer.
- MoE benefits depend strongly on distributed systems implementation.
- Expert routing introduces new failure modes.
- Token dropping can make training behavior harder to reason about.
- Parameter count can be misleading without active compute and throughput.
- Sparse capacity may be unnecessary or harmful in small-data regimes.

## Connections

- [[concepts/architectures/mixture-of-experts|Mixture of experts]]
- [[concepts/architectures/gating|Gating]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/systems/distributed-training-runbook|Distributed training]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/gpt-2|GPT-2]]
- [[papers/architectures/index|Architecture papers]]
