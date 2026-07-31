---
title: End-To-End Memory Networks
aliases:
  - papers/end-to-end-memory-networks
  - papers/memory-networks
tags:
  - papers
  - architectures
  - recurrent-models
  - memory
  - attention
---

# End-To-End Memory Networks

> A recurrent multi-hop attention architecture that reads an external memory several times before producing an answer.

## Metadata

| Field | Value |
| --- | --- |
| Paper | End-To-End Memory Networks |
| Authors | Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus |
| Year | 2015 |
| Venue | NeurIPS 2015 |
| arXiv | [1503.08895](https://arxiv.org/abs/1503.08895) |
| Status | verified |

## Question

An attention mechanism can retrieve one relevant memory in a single step. But many tasks require chained access: retrieve a fact, use it to form a better query, retrieve another fact, and only then answer. The paper asks whether repeated attention hops over an external memory can be trained end-to-end without requiring the model to expose intermediate reasoning labels.

The architecture separates:

- an input representation for the query;
- a memory representation for address keys;
- an output representation for values;
- a recurrent state updated after each hop;
- a final prediction over the answer vocabulary.

This is a compact precursor to architectures that repeatedly retrieve context before generation.

## Main Claim

The paper introduces a recurrent attention model over a possibly large external memory. Multiple computational hops improve performance on the reported question-answering tasks, while end-to-end training reduces the need for supervision about intermediate memory accesses.

The narrow claim is:

$$
\text{query}
\rightarrow
\text{attention read}
\rightarrow
\text{state update}
\rightarrow
\text{attention read}
\rightarrow
\text{answer}
$$

The model demonstrates learned multi-step retrieval under the paper's data and task protocols. It does not prove that attention weights are faithful explanations or that more hops always improve reasoning.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Query | encoded input question $q$ |
| Memory | a set or sequence of input facts $x_i$ |
| Address keys | memory embeddings $m_i$ |
| Values | output embeddings $c_i$ |
| Read | softmax attention over memory slots |
| State | query representation updated after each hop |
| Hops | repeated reads with shared or hop-specific parameters |
| Output | answer distribution over a vocabulary |

Let the query at hop $k$ be $u^k$. The memory contains input vectors $m_i$ and output vectors $c_i$. The attention probability for memory slot $i$ is:

$$
p_i^k
=
\operatorname{softmax}_i\left((u^k)^\top m_i\right)
=
\frac{\exp((u^k)^\top m_i)}
{\sum_j\exp((u^k)^\top m_j)}
$$

The retrieved output is:

$$
o^k
=
\sum_i p_i^k c_i
$$

and the next state is:

$$
u^{k+1}=u^k+o^k
$$

The residual update preserves the original query while adding evidence retrieved at hop $k$.

## Input and Output Embeddings

The model can use separate matrices for addressing and answering:

$$
m_i=A\Phi(x_i),
\qquad
c_i=C\Phi(x_i)
$$

where $\Phi$ converts a sentence or fact into a bag-of-words or positional representation, $A$ produces address keys, and $C$ produces value vectors. The query is embedded with:

$$
u^1=B\Phi(q)
$$

Using separate $A$ and $C$ is important. The representation that makes a fact easy to find need not be the same representation that should be returned after retrieval.

The final answer distribution can be:

$$
\hat{a}
=
\operatorname{softmax}(Wu^{K+1})
$$

where $W$ maps the final memory state to answer vocabulary logits.

## A Single Hop

For one hop, the data flow is:

1. encode the query as $u^1$;
2. encode each memory item as an address key $m_i$;
3. score every item with $(u^1)^\top m_i$;
4. normalize scores into $p_i^1$;
5. retrieve a weighted value vector $o^1$;
6. add the retrieved vector to the query state;
7. predict an answer from the updated state.

This is an attention read over an external memory. The memory can be a list of facts, sentences, or other discrete records; the architecture does not require those records to be part of the current hidden sequence in the same way as self-attention.

## Multiple Hops

The key architectural extension is repetition:

$$
u^1
\rightarrow
u^2
\rightarrow
\cdots
\rightarrow
u^{K+1}
$$

At each hop, the updated state changes the attention scores:

$$
p_i^{k+1}
=
\operatorname{softmax}_i\left((u^{k+1})^\top m_i\right)
$$

The first hop may retrieve a subject or relation, while the second hop uses that result to find another supporting fact. The paper's “hop” is a computational step, not automatically a human-interpretable reasoning step.

Parameter tying can make every hop use the same embedding and output matrices:

$$
A^1=\cdots=A^K,
\qquad
C^1=\cdots=C^K
$$

or different matrices can be used at different hops. Tying reduces parameters and encourages a repeated retrieval operation; untied hops permit stage-specific transformations but make capacity and interpretation different.

## Temporal and Positional Encoding

If a memory item is a sentence or sequence, a bag-of-words representation loses word order. A positional encoding can weight token embeddings by their position:

$$
\Phi(x)
=
\sum_{j=1}^{T}l_j\odot e_{x_j}
$$

where $e_{x_j}$ is the embedding of token $x_j$ and $l_j$ is a position-dependent vector. This allows the memory representation to preserve some local order without using a full recurrent encoder.

Temporal encoding can also distinguish memory slots that have similar content but occur at different positions. The general lesson is that an external memory still needs an explicit representation contract: “retrieve a sentence” is not enough if order or provenance affects the answer.

## Supervision Boundary

The model is trained from the final answer loss:

$$
\mathcal{L}
=
-\log p_\theta(y\mid q,x_{1:N})
$$

Gradients flow through the output layer, every hop, every attention distribution, and the memory embeddings. Intermediate supporting facts do not need to be labeled for the architecture to learn useful retrieval patterns on the task distribution.

This is a useful distinction:

| Training signal | What it can teach |
| --- | --- |
| final answer only | retrieval paths that help answer loss |
| answer plus supporting facts | explicit evidence selection constraints |
| answer plus hop supervision | more controlled intermediate access |
| contrastive memory objective | separation of relevant and irrelevant slots |

End-to-end optimization reduces annotation requirements, but it does not guarantee that every high-attention slot is causally sufficient evidence.

## Relation to RNNsearch and Transformers

The paper describes multiple hops as an extension of attention-based sequence models. The comparison is useful:

| Architecture | Memory source | Number of reads | State update |
| --- | --- | --- | --- |
| RNN encoder-decoder with attention | encoded source sequence | usually one read per decoder step | recurrent decoder state |
| End-to-End Memory Network | external memory records | multiple learned hops | residual query update |
| Transformer self-attention | current token states | one attention sublayer per block | residual stream across layers |
| RAG-style system | retrieved documents | external retriever and generation reads | discrete/system-level context update |

The important axis is not only “attention or no attention.” It is where the memory comes from, how many access steps occur, and whether the memory can be written or only read.

## Evidence

| Task | What it probes | Paper-level reading |
| --- | --- | --- |
| synthetic question answering | multi-fact retrieval and composition | multiple hops can improve answer accuracy |
| language modeling | use of memory reads in next-token prediction | memory architecture can be used beyond QA |
| varying number of hops | computational depth of retrieval | extra hops are useful when the task requires chained access |
| comparison to Memory Networks | end-to-end versus intermediate supervision | final-answer training can learn useful access patterns |

The results support the multi-hop retrieval design under the reported benchmarks. They do not establish that attention maps are complete explanations of model reasoning.

## Ablation and Reading Questions

| Question | What it isolates |
| --- | --- |
| Does performance improve from one hop to several? | value of iterative retrieval |
| Are embeddings tied across hops? | repeated operator versus stage-specific computation |
| Is positional encoding enabled? | order sensitivity in memory representation |
| Are memory slots fixed or dynamically updated? | read-only knowledge base versus writable state |
| Is the answer copied from memory or generated from state? | retrieval versus synthesis boundary |
| Are supporting facts labeled? | end-to-end learning versus explicit evidence supervision |

When reproducing the model, match the number of hops, memory size, vocabulary, answer candidates, positional encoding, parameter tying, and random seed. A change in any of these can alter the effective reasoning depth or capacity.

## Complexity and Systems Trade-offs

For $N$ memory slots, embedding dimension $d$, and $K$ hops, dense memory lookup costs approximately:

$$
O(KNd)
$$

per query, excluding the cost of encoding the memory and output projection. More hops increase computational depth and may improve compositional retrieval, but they also increase latency and create more opportunities for diffuse or incorrect attention.

| Scaling axis | Benefit | Cost or risk |
| --- | --- | --- |
| memory slots $N$ | larger searchable context | more score computation and distractors |
| hops $K$ | deeper retrieval composition | latency, instability, and over-retrieval |
| embedding dimension $d$ | richer keys and values | memory bandwidth and parameter count |
| separate value embeddings | decoupled address/value roles | additional capacity and mismatch risk |
| parameter tying | lower parameter count | less hop-specific specialization |

An external memory can be large in logical size while still being expensive to encode, move, and score. “Large memory” is not synonymous with cheap retrieval.

## Limitations

- Soft attention over a memory can mix multiple facts rather than retrieve a discrete record.
- Multiple hops do not guarantee faithful or logically valid reasoning.
- A final-answer objective may reward shortcut retrieval paths that correlate with the label.
- The architecture is primarily read-oriented; persistence and write governance require additional components.
- Synthetic QA tasks may not represent noisy documents, contradictory sources, or real retrieval distributions.
- Scaling the memory increases both compute and the number of distractor slots.
- The model's memory representation and answer vocabulary constrain what can be retrieved or generated.
- Later RAG systems add document retrieval, chunking, provenance, and generation policies that are outside this neural block.

## Relation to Other Architecture Papers

| Paper | Connection |
| --- | --- |
| [Memory Networks](/papers/architectures/memory-networks) | original memory-plus-inference formulation for QA |
| [Neural Turing Machines](/papers/architectures/neural-turing-machines) | differentiable external memory with explicit writes |
| [Differentiable Neural Computer](/papers/architectures/differentiable-neural-computer) | memory allocation and temporal-link traversal |
| [Hopfield Networks is All You Need](/papers/architectures/hopfield-networks-is-all-you-need) | one-step associative retrieval and attention correspondence |
| [Attention Is All You Need](/papers/architectures/attention-is-all-you-need) | attention as the dominant token-mixing operation |
| [Transformer-XL](/papers/architectures/transformer-xl) | recurrent segment-level context for longer sequences |

## Implementation Checklist

- Define whether the memory is read-only, append-only, or writable.
- Separate address embeddings from value embeddings when the roles differ.
- Log attention entropy and top-k memory slots at every hop.
- Test one-hop and multi-hop variants under matched compute.
- Check whether supporting facts remain selected under paraphrase and distractors.
- Evaluate memory size and hop count outside the training range.
- Record parameter tying and positional encoding as part of the architecture configuration.
- Treat retrieved evidence and final generated output as separate artifacts for verification.

## Why It Matters

End-To-End Memory Networks make “reasoning by repeated retrieval” an explicit neural block:

$$
\text{query}
\rightarrow
\text{read}
\rightarrow
\text{update query}
\rightarrow
\text{read again}
\rightarrow
\text{answer}
$$

This is a useful bridge from early external-memory networks to modern retrieval-augmented systems and agent workflows. It also gives the wiki a precise vocabulary for separating a model's internal attention context from a system's external knowledge store.

## Connections

- [[papers/architectures/neural-turing-machines|Neural Turing Machines]]
- [[papers/architectures/differentiable-neural-computer|Differentiable Neural Computer]]
- [[papers/architectures/hopfield-networks-is-all-you-need|Hopfield Networks is All You Need]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[agents/core/agent-memory|Agent memory]]
- [[papers/architectures/index|Architecture papers]]

---
