---
title: Neural Turing Machines
aliases:
  - papers/neural-turing-machines
  - papers/ntm
tags:
  - papers
  - architectures
  - recurrent-models
  - memory
---

# Neural Turing Machines

> A recurrent controller coupled to an external memory matrix through differentiable read and write operations.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Neural Turing Machines |
| Authors | Alex Graves, Greg Wayne, Ivo Danihelka |
| Year | 2014 |
| Venue | arXiv preprint |
| arXiv | [1410.5401](https://arxiv.org/abs/1410.5401) |
| Status | verified |

## Question

An ordinary [[concepts/architectures/rnn|RNN]] stores information in a fixed-size hidden state. That state must simultaneously represent the current computation, the history needed for later steps, and any temporary workspace. The paper asks whether a neural network can separate these roles by giving a recurrent controller an addressable external memory.

The design target is not simply a larger hidden state. It is a differentiable analogue of a computer with:

- a controller that transforms observations into actions;
- a memory matrix that stores a sequence of vectors;
- a read head that retrieves content;
- a write head that modifies memory;
- soft attention weights that keep the whole operation differentiable.

This is an architecture question about **where state lives**. The recurrent controller remains sequential, while the memory is an explicit data structure with a larger and more structured capacity.

## Main Claim

The paper argues that a neural network can learn simple algorithms such as copying, sorting, and associative recall when a recurrent controller is coupled to external memory with attention-based read and write operations.

The narrow claim is:

$$
\text{controller}
+
\text{differentiable memory access}
\Rightarrow
\text{learnable algorithmic behavior on the reported synthetic tasks}
$$

This does not establish reliable arbitrary computation, general-purpose reasoning, or deployment-quality long-context memory. The evidence is primarily a proof of architectural possibility on deliberately structured tasks.

## Architecture Contract

| Item | Contract |
| --- | --- |
| Input | observation sequence $x_1,\ldots,x_T$ |
| Controller | recurrent neural network |
| Memory | matrix $M_t \in \mathbb{R}^{N \times W}$ with $N$ locations and width $W$ |
| Interface | differentiable read and write heads |
| Read output | weighted sum of memory rows |
| Write output | erase then add update to memory rows |
| Training | gradient descent through the complete controller-memory computation |
| Output | controller output and/or task-specific prediction |

At time $t$, the controller receives an observation and the previous read vector:

$$
(h_t,\,\xi_t)
=
f_\theta(x_t, h_{t-1}, r_{t-1})
$$

where $h_t$ is the controller state, $\xi_t$ contains interface parameters, and $r_{t-1}$ is the previous read result. The controller does not directly manipulate a discrete address. It emits continuous parameters that determine a soft access distribution.

## Memory Representation

The external memory is a collection of row vectors:

$$
M_t
=
\begin{bmatrix}
m_t(1)^\top \\
\vdots \\
m_t(N)^\top
\end{bmatrix}
\in \mathbb{R}^{N \times W}
$$

The memory size is separated into two axes:

| Axis | Meaning |
| --- | --- |
| $N$ | number of addressable locations |
| $W$ | width of each stored vector |

This separation is important. Increasing the controller hidden dimension changes the computation state, while increasing $N$ changes how many slots can be addressed. The two capacities are not interchangeable.

## Addressing Mechanisms

Each head produces a weight vector $w_t \in \mathbb{R}^N$ satisfying:

$$
w_t(i) \ge 0,
\qquad
\sum_{i=1}^{N} w_t(i)=1
$$

The paper combines content-based addressing with location-based addressing.

### Content Addressing

For a key $k_t \in \mathbb{R}^W$, content similarity compares the key with each memory row. A cosine similarity is:

$$
K(k_t,m_t(i))
=
\frac{k_t \cdot m_t(i)}
{\lVert k_t\rVert_2\lVert m_t(i)\rVert_2}
$$

The content weights are obtained with a temperature or strength parameter $\beta_t$:

$$
w_t^c(i)
=
\frac{\exp\left(\beta_t K(k_t,m_t(i))\right)}
{\sum_{j=1}^{N}\exp\left(\beta_t K(k_t,m_t(j))\right)}
$$

Large $\beta_t$ makes the distribution sharper; small $\beta_t$ makes retrieval more diffuse. This is a soft form of content-addressable memory.

### Location Addressing

Content matching alone cannot express operations such as “move one slot to the right.” The location path therefore transforms a previous weighting using a learned interpolation, circular shift, and sharpening step.

First interpolate content and previous weights:

$$
w_t^g
=
g_t w_t^c + (1-g_t)w_{t-1}
$$

where $g_t \in [0,1]$ controls how much the head follows the new content match. A circular convolution shifts the weighting:

$$
\tilde{w}_t(i)
=
\sum_{j=0}^{N-1} w_t^g(j)s_t(i-j)
$$

where $s_t$ is a normalized shift distribution. Finally, sharpening changes the concentration:

$$
w_t(i)
=
\frac{\tilde{w}_t(i)^{\gamma_t}}
{\sum_j \tilde{w}_t(j)^{\gamma_t}}
$$

with $\gamma_t \ge 1$. The address is therefore a differentiable composition of content lookup and relative movement.

## Read Operation

Given a read weighting $w_t^r$, the read vector is the weighted sum of memory rows:

$$
r_t
=
\sum_{i=1}^{N}w_t^r(i)m_t(i)
$$

In matrix notation:

$$
r_t
=
(w_t^r)^\top M_t
\in \mathbb{R}^{W}
$$

The operation is differentiable with respect to both the weights and the memory contents. A useful consequence is that gradients can teach the controller not only what to store, but also which addressing pattern retrieves it later.

## Write Operation

Writing uses an erase vector $e_t \in [0,1]^W$ and an add vector $a_t \in \mathbb{R}^W$. The erase step is:

$$
\tilde{M}_t(i)
=
m_{t-1}(i)\odot
\left(\mathbf{1}-w_t^w(i)e_t\right)
$$

The add step is:

$$
m_t(i)
=
\tilde{M}_t(i)+w_t^w(i)a_t
$$

where $w_t^w$ is the write weighting and $\odot$ is element-wise multiplication. The factor $w_t^w(i)$ distributes the operation over locations, so a hard discrete write is replaced by a smooth write that can receive gradient signal.

The erase-add decomposition makes overwrite behavior explicit:

$$
M_{t-1}
\xrightarrow{\text{erase}}
\tilde{M}_t
\xrightarrow{\text{add}}
M_t
$$

## Controller and Memory Loop

One recurrent step can be read as:

1. receive the current input and the previous read vector;
2. update the controller state;
3. emit interface parameters;
4. compute read and write address weights;
5. write to memory;
6. read the new memory contents;
7. expose the read vector to the controller and output head.

The computation is sequential across $t$, but the memory interface is differentiable inside each step. This gives the model two distinct time scales: controller recurrence over steps and addressable storage over locations.

## Why This Is an Architecture Paper

The paper changes the primitive state transition from:

$$
h_t=f_\theta(h_{t-1},x_t)
$$

to:

$$
(h_t,M_t,r_t)
=
F_\theta(h_{t-1},M_{t-1},r_{t-1},x_t)
$$

The memory is no longer an implicit by-product of a fixed hidden vector. It is an explicit module with an interface contract. That pattern reappears in differentiable data structures, retrieval systems, external-memory agents, and tool-using systems.

## Evidence

| Task family | What it probes | Interpretation |
| --- | --- | --- |
| Copy | write a sequence and reproduce it later | separable storage and retrieval |
| Repeat copy | retain and emit a sequence multiple times | read timing and controller state |
| Associative recall | retrieve an item using a related cue | content addressing |
| Priority sort | store items and emit them in learned order | interaction between memory and controller |
| Simple sequence tasks | use input/output examples to infer a procedure | algorithm-like behavior under the tested distribution |

The results should be read as architectural demonstrations. Synthetic tasks make the intended algorithmic structure visible, but they do not measure broad natural-language or scientific generalization.

## Ablation and Reading Questions

| Question | Why it matters |
| --- | --- |
| Is content addressing necessary? | tests whether the model can retrieve by key rather than only by position |
| Is location addressing necessary? | tests sequential traversal and relative movement |
| Does memory size scale independently from controller size? | tests whether external storage creates a distinct capacity axis |
| Are tasks evaluated beyond training lengths? | separates memorization of sequence lengths from algorithmic extrapolation |
| Is the write distribution sharp or diffuse? | reveals whether the learned operation behaves like a discrete write |
| Is the controller recurrent state enough by itself? | checks whether the external memory provides measurable benefit |

For an implementation comparison, keep the following fixed: controller capacity, memory dimensions, training steps, task length distribution, initialization, and evaluation lengths. Otherwise an apparent memory benefit may be a capacity or optimization difference.

## Complexity and Systems Trade-offs

For a memory with $N$ locations and width $W$, a dense content lookup is approximately:

$$
O(NW)
$$

per head and step. The memory interface is therefore not free. It exchanges the compact hidden-state bottleneck of an RNN for explicit storage and address computation.

| Property | Ordinary RNN | NTM-style model |
| --- | --- | --- |
| persistent state | hidden vector | hidden vector plus memory matrix |
| access | implicit transition | differentiable read/write |
| capacity knob | hidden width | memory slots and width |
| long-range path | repeated recurrent updates | repeated updates plus address reuse |
| parallelism across time | limited | still limited by controller loop |
| implementation risk | hidden-state stability | addressing, write interference, memory initialization |

The NTM does not solve sequential latency. It changes the representation and access path of state.

## Limitations

- Soft addressing can blur writes across multiple locations and create interference.
- The controller still processes time steps recurrently, so training and inference retain sequential dependencies.
- Synthetic algorithmic tasks are not evidence for general reasoning or reliable program induction.
- Addressing distributions can be difficult to optimize, especially when a task needs nearly discrete pointer behavior.
- Memory size and initialization are part of the effective model; changing them changes the task capacity.
- A read vector is a weighted average, so exact retrieval can degrade when keys are similar or memory contents are noisy.
- The architecture does not define persistence, provenance, or access control for real-world knowledge.
- External memory does not by itself make a model factual, grounded, or safe.

## Relation to Later Architectures

| Later idea | Shared problem | Difference |
| --- | --- | --- |
| [LSTM](/papers/architectures/long-short-term-memory) | preserve information across steps | LSTM stores state in a gated fixed-size vector |
| [Transformer](/papers/architectures/attention-is-all-you-need) | select relevant context | Transformer performs token-to-token attention without an explicit writable memory matrix |
| [Perceiver IO](/papers/architectures/perceiver-io) | route information through a bottleneck | Perceiver uses learned latent arrays and cross-attention rather than NTM write/erase dynamics |
| [agent memory](/agents/core/agent-memory) | retain and retrieve information across actions | agent memory adds persistence, provenance, and workflow policy outside the neural block |
| [Mamba](/papers/architectures/mamba) | maintain a compact sequence state | selective state-space updates a recurrent state with input-dependent dynamics |

The useful distinction is not “memory versus no memory.” Every sequence model has state somewhere. The distinction is whether state is an explicit addressable data structure, a fixed recurrent state, an attention context, or a system-level store.

## Implementation Checklist

- State the memory shape $(N,W)$ separately from controller hidden size.
- Verify that read and write weights are normalized and non-negative.
- Keep erase and add operations numerically stable and shape-checked.
- Log address entropy or concentration when diagnosing diffuse reads and writes.
- Test copy and associative recall with lengths outside the training range.
- Compare against a controller-only baseline with matched parameter and training budgets.
- Treat initialization as part of the reproducibility contract.
- Separate in-memory state from persistent external artifacts when building a real system.

## Why It Matters

NTM is a canonical paper for the question “what should a model remember, and how should it address that memory?” It makes the controller-memory boundary explicit before later systems rediscover variants of the same boundary through attention, retrieval, latent arrays, or agent memory.

The reusable pattern is:

$$
\text{observation}
\rightarrow
\text{controller}
\rightarrow
\text{soft address}
\rightarrow
\text{read/write memory}
\rightarrow
\text{next state and output}
$$

For this wiki, read it after [[papers/architectures/long-short-term-memory|LSTM]] and before [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]] when studying how sequence models represent and access history.

## Connections

- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[agents/core/agent-memory|Agent memory]]
- [[papers/architectures/long-short-term-memory|LSTM]]
- [[papers/architectures/perceiver-io|Perceiver IO]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
- [[papers/architectures/index|Architecture papers]]

---
