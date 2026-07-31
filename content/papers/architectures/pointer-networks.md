---
title: Pointer Networks
aliases:
  - papers/pointer-networks
  - papers/ptr-net
tags:
  - papers
  - architectures
  - attention
  - structured-prediction
---

# Pointer Networks

> An attention-based sequence model whose output symbols are positions in the input sequence rather than members of a fixed vocabulary.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Pointer Networks |
| Authors | Oriol Vinyals, Meire Fortunato, Navdeep Jaitly |
| Year | 2015 |
| Venue | NeurIPS 2015 |
| arXiv | [1506.03134](https://arxiv.org/abs/1506.03134) |
| Status | verified |

## Question

Ordinary sequence-to-sequence models predict tokens from a fixed output dictionary. This breaks down when the output dictionary itself depends on the input length. Sorting a variable-length list, selecting a route through a graph, or returning the vertices of a convex hull requires output symbols that are input positions.

Pointer Networks change the role of attention. In an ordinary encoder-decoder model, attention produces a context vector by mixing encoder states. In a pointer network, attention probabilities are themselves the output distribution over input positions.

The architecture question is:

$$
\text{fixed vocabulary output}
\quad\longrightarrow\quad
\text{input-dependent pointer output}
$$

## Main Claim

The paper introduces a neural architecture that learns conditional distributions over positions in a variable-length input sequence. It applies the model to sorting and geometric combinatorial problems, including convex hull, Delaunay triangulation, and planar travelling-salesperson tasks.

The bounded claim is:

$$
p(y_t\mid y_{<t},x_{1:n})
\text{ can be defined over }\{1,\ldots,n\}
$$

This is an output-space contribution. It does not claim that the model exactly solves every combinatorial problem, nor that attention weights are always faithful explanations of the predicted structure.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | variable-length sequence $x_{1:n}$ |
| Encoder | recurrent encoder producing one state per input position |
| Decoder | recurrent autoregressive decoder |
| Attention | scores every encoder position at each output step |
| Output space | input indices $1,\ldots,n$ plus task-specific stop handling |
| Output | permutation, subset, or sequence of pointers |
| Training | teacher-forced cross-entropy over target indices |

The output vocabulary is not a fixed set of words. It is the current input index set:

$$
\mathcal{Y}(x_{1:n})
=
\{1,2,\ldots,n\}
$$

This makes the output dimension depend on the example.

## Encoder and Decoder States

An encoder maps each input item to a contextual state:

$$
h_i=\operatorname{Encoder}(x_i,h_{i-1}),
\qquad i=1,\ldots,n
$$

The decoder maintains an autoregressive state:

$$
s_t
=
\operatorname{Decoder}(y_{t-1},s_{t-1},c_{t-1})
$$

where $y_{t-1}$ is the previously selected input position and $c_{t-1}$ is the previous context. The model can therefore condition the next pointer on both the entire encoded input and the partial output structure.

## Pointer Attention

For decoder state $s_t$ and encoder state $h_i$, an additive attention score can be written as:

$$
u_i^t
=
v^\top
\tanh(W_hh_i+W_ss_t)
$$

The pointer distribution is:

$$
p(y_t=i\mid y_{<t},x)
=
\frac{\exp(u_i^t)}
{\sum_{j=1}^{n}\exp(u_j^t)}
$$

The selected pointer can be sampled, decoded greedily, or chosen by beam search. The context vector is still useful for updating the decoder:

$$
c_t=\sum_{i=1}^{n}p(y_t=i\mid y_{<t},x)h_i
$$

But the main output is the index distribution itself, not merely $c_t$.

## Difference from Ordinary Attention

| Mechanism | Attention output | Output distribution |
| --- | --- | --- |
| encoder-decoder attention | weighted context vector | generated token vocabulary |
| self-attention | updated token representations | task head or next-token vocabulary |
| pointer network | weighted context plus pointer scores | input positions |
| copy mechanism | mixture of vocabulary and source tokens | vocabulary and/or source positions |

The pointer network makes the attention-to-output connection explicit. This is useful when the answer must refer to an entity already present in the input rather than generate a new symbol.

## Autoregressive Objective

For a target pointer sequence $y_{1:T}$:

$$
\mathcal{L}_{\text{ptr}}
=
-\sum_{t=1}^{T}
\log p_\theta(y_t\mid y_{<t},x_{1:n})
$$

Teacher forcing supplies the gold previous pointer during training. At inference, the model consumes its own selected positions:

$$
\hat{y}_t
=
\arg\max_{i\in\mathcal{Y}(x)}
p_\theta(y_t=i\mid\hat{y}_{<t},x)
$$

This creates exposure bias and makes constraint handling part of decoding.

## Structured Output Constraints

For a permutation task, an input position should normally be selected at most once. A decoder can enforce this with a mask:

$$
u_i^t
\leftarrow
\begin{cases}
u_i^t & i\notin V_{t-1} \\
-\infty & i\in V_{t-1}
\end{cases}
$$

where $V_{t-1}$ is the set of previously visited positions. The normalized distribution is then defined only over valid candidates.

This separates two ideas:

- the neural model scores candidate positions;
- the decoder enforces task legality.

If legality is not encoded, a high-probability output can still be an invalid permutation or graph structure.

## Variable Output Dictionaries

Suppose a fixed-vocabulary decoder has $V$ output classes. Its final logits have shape:

$$
z_t\in\mathbb{R}^{V}
$$

For a pointer network, the logits have shape:

$$
z_t\in\mathbb{R}^{n}
$$

where $n$ is the current input length. The output head is therefore coupled to the encoder sequence, not to a global vocabulary matrix.

This is the central implementation detail. Batching variable-length examples requires padding or packed representations, but padded positions must be masked before softmax so that they cannot become valid pointers.

## Geometric and Combinatorial Tasks

The paper applies pointer networks to problems whose output is a sequence of input elements:

| Task | Input | Output |
| --- | --- | --- |
| sorting | unordered or ordered values | positions in sorted order |
| convex hull | planar points | indices of hull vertices |
| Delaunay triangulation | planar points | tuples or sequences of point indices |
| planar TSP | points or graph nodes | a tour represented by input positions |

These tasks make the output-space issue obvious. A conventional softmax over a fixed vocabulary cannot naturally represent an arbitrary set of new point identities without an additional indexing mechanism.

## Approximate Algorithms and Constraints

The model produces a conditional distribution, not a proof of optimality:

$$
\hat{y}
=
\operatorname{Decode}(p_\theta(y_{1:T}\mid x))
$$

For combinatorial optimization, the decoder may produce a feasible but suboptimal solution. Evaluation should separate:

| Metric | Meaning |
| --- | --- |
| validity | output obeys permutation/graph constraints |
| objective value | cost of the predicted solution |
| optimality gap | difference from an optimal or reference solution |
| length generalization | behavior beyond training input sizes |
| decoding cost | runtime as candidate count and output length grow |

Accuracy of each pointer is not sufficient when the whole output is a structured object.

## Evidence

| Claim | Evidence type | Boundary |
| --- | --- | --- |
| variable-size output dictionaries are learnable | pointer distributions over input positions | depends on input representation and decoder |
| attention can directly select input members | index-valued output probabilities | not the same as faithful explanation |
| learned models can approximate combinatorial procedures | geometric task experiments | approximate solution quality is task- and size-dependent |
| length generalization is possible | evaluation beyond some training lengths | does not imply arbitrary extrapolation |

The paper is most important for the output contract, not for claiming that a recurrent model replaces exact combinatorial solvers.

## Ablation and Reading Questions

| Question | What it isolates |
| --- | --- |
| Are pointer logits masked for invalid positions? | output legality versus raw scoring |
| Is the output a permutation or can positions repeat? | task constraint contract |
| Does training include larger inputs than evaluation? | genuine length generalization versus interpolation |
| Is decoding greedy or beam search? | model score versus search procedure |
| Are encoder and decoder capacities matched to baselines? | architecture benefit versus parameter budget |
| Is objective gap reported alongside accuracy? | feasible output quality versus token-level agreement |

For a reproduction, record padding behavior, stop token, visited-position mask, teacher forcing policy, beam width, and reference solver used to compute objective gaps.

## Complexity and Systems Trade-offs

At decoder step $t$, scoring all $n$ encoder positions costs approximately:

$$
O(nd)
$$

for hidden dimension $d$. For output length $T$:

$$
O(Tnd)
$$

plus encoder and decoder costs. This is similar to cross-attention in an autoregressive decoder, but the logits now directly define the output candidates.

| Property | Benefit | Cost or risk |
| --- | --- | --- |
| input-dependent output space | handles arbitrary entity identities | batching and masking are more complex |
| attention as pointer | simple candidate scoring | dense candidate scan per output step |
| autoregressive decoding | models output dependencies | sequential latency and exposure bias |
| hard constraints | valid structured outputs | constraint implementation can dominate behavior |
| variable-length generalization | can extrapolate candidate count | not guaranteed outside the training distribution |

## Relation to Later Architectures

| Later idea | Connection |
| --- | --- |
| copy and pointer-generator models | mix vocabulary generation with source-position copying |
| [NTM](/papers/architectures/neural-turing-machines) | also uses attention over input/memory, but its output is not inherently an input index |
| [Transformer](/papers/architectures/attention-is-all-you-need) | provides the attention block that can be adapted to pointer-style output heads |
| [Set Transformer](/papers/architectures/set-transformer) | models permutation-invariant sets but does not by itself define an index-valued output |
| [Graphormer](/papers/architectures/graphormer) | adds structural graph bias while retaining token-style outputs |
| tool and agent selection | selecting a tool, file, or candidate can use a pointer-like index space plus validation |

The reusable pattern is “score candidates that already exist.” It can be applied to molecules, graph nodes, retrieved documents, tools, or actions, but each domain needs its own validity and provenance contract.

## Limitations

- Pointer probabilities are not automatically explanations or certificates.
- Autoregressive decoding can accumulate early selection errors.
- A pointer model may learn heuristics rather than the intended exact algorithm.
- Length extrapolation is empirical and can fail abruptly.
- Dense candidate scoring becomes expensive for very long inputs.
- Constraint masks can hide model errors by repairing or forcing outputs.
- Objective quality for combinatorial tasks requires task-specific evaluation, not only cross-entropy.

## Implementation Checklist

- Define the candidate index space for every example.
- Mask padding, visited, illegal, and unavailable candidates before softmax.
- Keep pointer labels tied to stable input identifiers through preprocessing.
- Record whether the model points to raw input, encoded chunks, graph nodes, or retrieved documents.
- Evaluate validity, objective gap, and length generalization separately.
- Compare greedy, beam, and constraint-aware decoding under fixed compute.
- Test duplicate or near-duplicate candidates and verify tie behavior.
- Preserve source position and provenance when a pointer is passed to another module.

## Why It Matters

Pointer Networks establish a clean output-space principle:

$$
\text{attention scores over input positions}
=
\text{output distribution over input entities}
$$

That principle is important for a broad AI wiki because many systems do not generate a new object. They select a protein residue, molecule, graph node, retrieved passage, tool, action, or candidate structure from a variable-sized set.

Read it after [[papers/architectures/neural-machine-translation-align-translate|Bahdanau attention]] and before structured retrieval or graph decision models.

## Connections

- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder architectures]]
- [[papers/architectures/neural-machine-translation-align-translate|Bahdanau attention]]
- [[papers/architectures/neural-turing-machines|Neural Turing Machines]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/graphormer|Graphormer]]
- [[papers/architectures/index|Architecture papers]]

---
