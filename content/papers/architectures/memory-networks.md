---
title: Memory Networks
aliases:
  - papers/memory-networks
  - papers/memory-network
tags:
  - papers
  - architectures
  - memory
  - question-answering
---

# Memory Networks

> A neural architecture that separates inference from a readable and writable long-term memory.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Memory Networks |
| Authors | Jason Weston, Sumit Chopra, Antoine Bordes |
| Year | 2014 |
| Venue | arXiv preprint |
| arXiv | [1410.3916](https://arxiv.org/abs/1410.3916) |
| Status | verified |

## Question

Standard neural networks often compress the information needed for a prediction into a fixed hidden state. That is a poor fit for tasks where the relevant facts are stored in a changing collection of sentences or records and an answer requires chaining several facts together.

Memory Networks propose a system-level decomposition:

- an input module that converts observations into internal representations;
- a generalization module that updates the current state;
- an output module that produces a query for memory access;
- a response module that maps the final state to an answer;
- a long-term memory that can be read and written.

The key design decision is to make memory a named component rather than treating it as an undocumented part of the hidden state.

## Main Claim

The paper introduces a class of models that combine inference components with a long-term memory component. It evaluates the design on question answering where memory acts as a dynamic knowledge base.

The bounded claim is:

$$
\text{inference state}
+
\text{long-term memory}
\rightarrow
\text{multi-step prediction}
$$

The paper provides a framework for learning how to use memory jointly with inference. It does not establish that a memory read is automatically a proof, that the selected facts are the only sufficient evidence, or that the same access policy transfers to arbitrary documents.

## Architecture Contract

| Component | Symbol | Role |
| --- | --- | --- |
| input feature map | $I$ | maps raw input to an internal representation |
| generalization | $G$ | updates the current internal state |
| output feature map | $O$ | produces a memory query or output representation |
| response | $R$ | maps the final state to an answer |
| memory | $m_i$ | stores facts or records accessible by the model |

For a query $q$, the system can be summarized as:

$$
u_1=I(q),
\qquad
u_{k+1}=G(u_k,m_{i_k}),
\qquad
y=R(u_K)
$$

where the memory index $i_k$ is selected by an output or addressing operation. The original formulation allows the components to be implemented in different ways; the architecture is a contract, not one fixed neural block.

## Memory as a Knowledge Base

Let the memory contain records:

$$
\mathcal{M}=\{m_1,m_2,\ldots,m_N\}
$$

Each record may be a sentence, fact, entity description, or other representation. The memory can be dynamic: new observations may be added, old records may be modified, and the system can learn how to use the collection for a prediction.

This gives the model a different capacity axis from the inference network:

| Capacity | Controlled by |
| --- | --- |
| representation and computation | parameters of $I$, $G$, $O$, and $R$ |
| number of facts | memory size $N$ |
| fact representation | encoding and storage policy |
| reasoning depth | number of memory access or update steps |
| knowledge freshness | write, replacement, and provenance policy |

The last row is a system property. A neural memory can be writable without being trustworthy or auditable.

## Addressing and Inference

An output module produces an internal query:

$$
q_k=O(u_k)
$$

An addressing function scores memory records:

$$
s_i^k=\operatorname{score}(q_k,m_i)
$$

For a differentiable implementation, normalized access weights are:

$$
p_i^k
=
\frac{\exp(s_i^k)}{\sum_j\exp(s_j^k)}
$$

and a soft memory read is:

$$
r_k=\sum_i p_i^k v_i
$$

where $v_i$ is the value representation associated with record $m_i$. A discrete memory network may instead select or rank records using a non-differentiable or supervised access mechanism.

The distinction between key and value is important:

| Representation | Purpose |
| --- | --- |
| key | decide which record matches the current query |
| value | provide information to the inference state |
| provenance | identify source, version, or evidence context |
| write metadata | decide whether and how the record may change |

The original architecture primarily establishes the first two. Practical public LLM systems need the latter two as explicit contracts.

## Multiple Supporting Facts

Suppose the answer requires facts $m_a$ and $m_b$. A multi-step system can first read $m_a$, update its state, and then use the updated state to access $m_b$:

$$
u_1=I(q)
$$

$$
r_1=\operatorname{Read}(O(u_1),\mathcal{M}),
\qquad
u_2=G(u_1,r_1)
$$

$$
r_2=\operatorname{Read}(O(u_2),\mathcal{M}),
\qquad
u_3=G(u_2,r_2)
$$

$$
\hat{y}=R(u_3)
$$

The updated query can make a second fact accessible. This is the architectural precursor to multi-hop retrieval; the number of hops is a computational depth choice, not automatically the number of human reasoning steps.

## Supervision and Stronger Memory Interfaces

The paper distinguishes the final prediction from the internal access process. A system may be trained with:

| Signal | What it constrains |
| --- | --- |
| final answer | whether the entire system predicts correctly |
| supporting facts | whether useful records are selected |
| access order | whether the intended inference route is followed |
| memory writes | which knowledge may persist or change |
| provenance | whether the answer can be traced to a source |

End-to-end answer training can be cheaper in annotation but may leave access behavior underconstrained. A model can arrive at the right answer through a shortcut, spurious correlation, or incomplete evidence.

## Relation to End-To-End Memory Networks

The original Memory Networks formulation makes the inference-memory decomposition explicit. [[papers/architectures/end-to-end-memory-networks|End-To-End Memory Networks]] turns the access process into a recurrent attention mechanism that can be trained with the final task loss.

| Design choice | Memory Networks | End-To-End Memory Networks |
| --- | --- | --- |
| memory access | explicit memory component and inference operations | differentiable attention over memory slots |
| intermediate supervision | can be used for access/inference components | reduced by end-to-end training |
| multiple hops | supported as repeated inference | central computational mechanism |
| output | task-dependent response module | answer distribution from final state |
| interpretation | modular system contract | learned soft retrieval trajectory |

## Relation to DNC and NTM

| Architecture | Memory behavior |
| --- | --- |
| [NTM](/papers/architectures/neural-turing-machines) | differentiable read/write over a matrix with content and location addressing |
| [DNC](/papers/architectures/differentiable-neural-computer) | adds allocation and temporal links for structured memory manipulation |
| Memory Networks | treats memory as a knowledge base for inference and response |
| End-To-End Memory Networks | performs multiple differentiable reads before prediction |
| [Modern Hopfield network](/papers/architectures/hopfield-networks-is-all-you-need) | focuses on content-addressed associative retrieval and fixed-point dynamics |

These models share a memory vocabulary but differ in what can be written, how addressing is trained, and whether the memory is a transient tensor or a persistent knowledge store.

## Evidence

| Evidence type | What it supports | What it does not prove |
| --- | --- | --- |
| question answering with a dynamic knowledge base | memory can participate in prediction | factual reliability on open-world data |
| chained supporting sentences | multi-step access can be useful | faithful reasoning trace |
| comparison with fixed-state models | explicit memory can add capacity | universal superiority |
| memory read/write behavior | modular access is learnable | safe persistence or provenance |

Read the paper as an architecture proposal and a set of QA experiments. Do not convert the existence of a memory module into a claim that the model has human-like long-term memory.

## Ablation and Reading Questions

| Question | Why it matters |
| --- | --- |
| Is the memory read-only or writable? | separates retrieval from continual storage |
| Is memory size fixed during training and evaluation? | tests capacity and extrapolation |
| Are facts independently encoded? | reveals whether addressing depends on context leakage |
| Are supporting facts supervised? | determines how much access behavior is constrained |
| Can the model answer with contradictory records? | tests source conflict handling |
| Does the response module copy or synthesize? | separates lookup from generation |

For a modern implementation, also record chunking, deduplication, indexing, source version, retrieval top-k, reranking, and whether the final answer is allowed to use information not returned by memory.

## Complexity and Systems Trade-offs

Dense memory scoring for $N$ records of dimension $d$ costs approximately:

$$
O(Nd)
$$

per access step. With $K$ hops:

$$
O(KNd)
$$

before accounting for encoding, indexing, reranking, and response generation.

| Scaling axis | Benefit | Risk |
| --- | --- | --- |
| more records | broader knowledge coverage | more distractors and higher lookup cost |
| more hops | more compositional access | latency and error accumulation |
| larger embeddings | richer key/value representations | memory bandwidth and overfitting |
| writable memory | freshness and adaptation | stale, conflicting, or unsafe writes |
| external index | faster candidate retrieval | approximate recall and index drift |

The original paper does not provide a complete production retrieval system. Modern systems need an index, document lifecycle, provenance, and verification layer around the neural access operation.

## Limitations

- A memory module does not guarantee that the selected facts are sufficient or correct.
- Soft or learned addressing can be diffuse and sensitive to distractors.
- Final-answer supervision may leave the memory access path underdetermined.
- A dynamic knowledge base requires write policy, versioning, conflict resolution, and access control.
- Synthetic or closed-world QA does not measure open-world retrieval, stale sources, or citation correctness.
- Increasing memory size can increase latency and introduce more false matches.
- A modular architecture does not automatically make each module independently interpretable.

## Relation to LLM and Agent Systems

The paper's decomposition maps cleanly onto modern systems, but the mapping must remain explicit:

| Memory Network concept | Modern analogue |
| --- | --- |
| long-term memory | document store, vector index, database, or durable file |
| output module | query encoder, retriever, or planner |
| generalization module | reranker, reader, or language-model state update |
| response module | answer generator or structured-output head |
| read/write policy | retrieval pipeline, tool contract, or memory lifecycle |

In an agent, the memory is not just another prompt string. It has provenance, permissions, freshness, and verification requirements. The neural architecture describes how information can be used; the surrounding workflow decides whether it may be trusted or persisted.

## Implementation Checklist

- Define the memory record schema and provenance fields.
- Separate retrieval keys, returned values, and source metadata.
- Log top-k candidates and scores for every access step.
- Evaluate with distractors, duplicates, contradictions, and stale records.
- Distinguish final-answer accuracy from evidence-selection accuracy.
- Specify whether writes are enabled, who authorizes them, and how versions are retained.
- Compare one-hop and multi-hop variants under matched compute.
- Keep the external memory boundary visible in the system diagram.

## Why It Matters

Memory Networks provide the cleanest early statement of a design principle that remains central to LLM and agent systems:

$$
\text{model parameters}
\ne
\text{long-term knowledge}
$$

Knowledge can live in a memory store and be accessed by a learned inference process. The important engineering questions then become what is stored, how it is addressed, how many access steps are allowed, and what evidence accompanies the answer.

Read it before [[papers/architectures/end-to-end-memory-networks|End-To-End Memory Networks]], then compare the result with [[papers/architectures/differentiable-neural-computer|DNC]] and modern retrieval-augmented generation.

## Connections

- [[papers/architectures/end-to-end-memory-networks|End-To-End Memory Networks]]
- [[papers/architectures/neural-turing-machines|Neural Turing Machines]]
- [[papers/architectures/differentiable-neural-computer|Differentiable Neural Computer]]
- [[papers/architectures/hopfield-networks-is-all-you-need|Hopfield Networks is All You Need]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[agents/core/agent-memory|Agent memory]]
- [[papers/architectures/index|Architecture papers]]

---
