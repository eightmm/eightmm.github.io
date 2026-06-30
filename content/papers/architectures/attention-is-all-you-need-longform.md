---
title: Attention Is All You Need Longform Review
aliases:
  - papers/attention-is-all-you-need-longform
  - papers/transformer-longform
tags:
  - papers
  - longform
  - architectures
  - transformer
  - attention
---

# Attention Is All You Need Longform Review

> A beginner-friendly longform review of the Transformer paper: why it mattered, how attention replaced recurrence in the tested sequence-transduction setting, what the evidence supports, and what later readers should not overclaim.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Attention Is All You Need |
| Authors | Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin |
| Year | 2017 |
| Venue | NeurIPS 2017 |
| arXiv | [1706.03762](https://arxiv.org/abs/1706.03762) |
| NeurIPS | [Proceedings page](https://papers.nips.cc/paper/7181-attention-is-all-you-need) |
| Compact note | [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]] |
| Reading status | verified |

## One-Line Summary

The paper introduced the [[concepts/architectures/transformer|Transformer]], an encoder-decoder architecture that uses attention as the main sequence-mixing operation and removes recurrent and convolutional sequence layers from the model core.

## Why This Paper Mattered

The paper is easy to misread because the title sounds like a slogan from the modern LLM era. In the original setting, the paper was about sequence transduction, especially machine translation. The core question was practical and architectural:

Can a model map one sequence to another without processing tokens recurrently or with convolutional layers?

Before the Transformer, strong neural machine translation systems often used encoder-decoder models built from recurrent networks such as LSTMs or GRUs. Attention already existed, but it was usually added to a recurrent encoder-decoder model so the decoder could look back at relevant source positions. The Transformer made a stronger architectural move: make attention the central mixing operation.

That matters because recurrence has an inherent sequential dependency. If token state $h_t$ depends on $h_{t-1}$, then the model cannot compute all positions in a layer independently:

$$
h_t = f(h_{t-1}, x_t)
$$

This dependency is useful as an inductive bias, but it limits parallelism over sequence length. The Transformer replaces this with attention, where token-to-token interactions in a layer can be computed with matrix operations:

$$
Y = \operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

The paper's lasting contribution is not only a new formula. It is the demonstration that a stack of attention and feed-forward blocks can be a complete high-performing sequence model.

## Background Before the Paper

The old sequence modeling story was built around order. A sentence is a sequence, so it felt natural to process it left-to-right or with local windows. Recurrent networks process tokens in order. Convolutional networks process local neighborhoods and grow receptive field with depth. Both encode useful assumptions:

| Architecture family | Main sequence bias | Practical issue |
| --- | --- | --- |
| RNN/LSTM/GRU | state evolves through time | limited parallelism across positions |
| CNN sequence model | local windows compose into larger context | long-range relation needs depth or dilation |
| Attention-augmented seq2seq | decoder can refer to source positions | attention is not the whole model |
| Transformer | every token can mix with other tokens in a layer | dense attention has quadratic token-pair cost |

The Transformer changes the default question. Instead of asking how to carry a hidden state through time, it asks which positions should exchange information.

This is why [[concepts/architectures/attention|Attention]] is the central concept. Attention turns a set of token states into new token states by computing query-key similarities and using them to mix values.

## The Main Idea

The paper's main idea can be summarized as:

$$
\text{sequence modeling}
\approx
\text{attention-based token mixing}
+
\text{position-wise nonlinear processing}
+
\text{positional information}
$$

The Transformer block alternates between two operations:

| Operation | Role |
| --- | --- |
| Multi-head attention | mixes information across token positions |
| Feed-forward network | transforms each token independently after mixing |

Residual connections and normalization make deep stacking trainable. Positional encodings restore order information that plain attention does not know by itself.

In other words, the model separates three concerns:

| Concern | Transformer component |
| --- | --- |
| Which tokens interact? | attention weights |
| What information is transferred? | value vectors |
| Where is the token in the sequence? | positional encoding |

## Scaled Dot-Product Attention

The paper defines scaled dot-product attention as:

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$

The symbol table is:

| Symbol | Meaning |
| --- | --- |
| $Q$ | query matrix |
| $K$ | key matrix |
| $V$ | value matrix |
| $d_k$ | key/query dimension |
| $QK^\top$ | all query-key dot products |
| $\sqrt{d_k}$ | scaling factor |
| softmax | normalizes scores over candidate keys |

For a single query $q_i$, the attention weight on key $k_j$ is:

$$
\alpha_{ij}
=
\frac{
\exp(q_i^\top k_j / \sqrt{d_k})
}{
\sum_{\ell}
\exp(q_i^\top k_\ell / \sqrt{d_k})
}
$$

The output for that query is a weighted sum of value vectors:

$$
y_i
=
\sum_j \alpha_{ij}v_j
$$

This looks simple, but it changes the computation graph. Every token can directly attend to every other token in a layer. That gives short information paths between distant tokens, at the cost of forming a token-pair score matrix.

## Why the Scaling Term Appears

The division by $\sqrt{d_k}$ is not decoration. If query and key components have roughly unit variance, their dot product tends to grow in scale with dimension. Large logits can push [[concepts/architectures/softmax|softmax]] into saturated regions where gradients become less useful.

The scaling term keeps the logits in a more stable range:

$$
\frac{q^\top k}{\sqrt{d_k}}
$$

This is a good example of how architecture papers mix modeling ideas and numerical training details. The attention mechanism is a modeling idea; the scale factor helps make it trainable in the chosen parameterization.

## Multi-Head Attention

Single attention can mix tokens through one set of projected features. Multi-head attention runs several attention operations in parallel:

$$
\operatorname{head}_i
=
\operatorname{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

$$
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W^O
$$

Each head has its own projection matrices. The point is not that each head has a guaranteed human-readable role. The point is that the model can represent multiple relation types or subspaces at the same layer.

For example, one head might learn a local syntactic relation, another might carry longer-range dependency information, and another might behave like a position-sensitive copying mechanism. But this is an interpretation, not a guarantee. [[concepts/architectures/attention|Attention]] weights are useful diagnostics, not automatic explanations.

## Encoder, Decoder, and Cross-Attention

The paper uses an encoder-decoder Transformer.

The encoder reads the source sequence. Each encoder layer has:

1. multi-head self-attention;
2. position-wise feed-forward network.

The decoder generates the target sequence. Each decoder layer has:

1. masked self-attention over previously generated target positions;
2. encoder-decoder attention over source representations;
3. position-wise feed-forward network.

The decoder self-attention is masked so position $t$ cannot look at future target tokens. In modern notation, the masked attention is:

$$
\operatorname{Attention}(Q,K,V,M)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)V
$$

where future positions receive a very negative mask value before softmax.

This distinction matters:

| Attention type | Query comes from | Key/value comes from | Purpose |
| --- | --- | --- | --- |
| encoder self-attention | source tokens | source tokens | source representation |
| decoder masked self-attention | target prefix | target prefix | autoregressive target state |
| encoder-decoder attention | target state | encoded source | condition generation on source |

This is why the paper belongs under [[papers/architectures/index|Architecture papers]] and not only under LLMs. The original architecture is an encoder-decoder sequence transduction model. Decoder-only LLMs are later descendants, not the exact same setting.

## Positional Encoding

Attention by itself does not know token order. If the input token states are permuted and no positional information is added, plain self-attention is permutation-equivariant. The paper injects order using sinusoidal positional encodings:

$$
\operatorname{PE}_{(pos,2i)}
=
\sin\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right)
$$

$$
\operatorname{PE}_{(pos,2i+1)}
=
\cos\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right)
$$

These vectors are added to token embeddings:

$$
x_t = e_t + \operatorname{PE}_t
$$

The important conceptual point is simple: the Transformer needs some representation of order because attention alone treats tokens as content vectors with pairwise interactions.

The paper also tested learned positional embeddings and reported similar results in the tested setting. The sinusoidal choice is often remembered because it is explicit and can be evaluated beyond the positions seen during training, though actual length extrapolation always needs empirical checking.

## Feed-Forward Blocks

Attention mixes information across positions, but the model also needs nonlinear transformation at each position. The paper uses a position-wise feed-forward network:

$$
\operatorname{FFN}(x)
=
\max(0,xW_1+b_1)W_2+b_2
$$

The same feed-forward network is applied independently to each position. In modern Transformer language, this is the MLP block.

So one simplified encoder layer is:

$$
X'
=
\operatorname{Norm}
\left(
X + \operatorname{MultiHead}(X,X,X)
\right)
$$

$$
Y
=
\operatorname{Norm}
\left(
X' + \operatorname{FFN}(X')
\right)
$$

The exact normalization placement in the original paper is now often called post-norm. Many later models use pre-norm variants for training stability. This is a good example of why the original paper should be read as a foundation, not as the final canonical implementation of all later Transformers.

## Complexity and Parallelism

The paper emphasizes that self-attention has shorter path length between positions and more parallelism than recurrent models. Dense self-attention forms a pairwise matrix:

$$
S = QK^\top
$$

For sequence length $T$, full self-attention has memory proportional to:

$$
O(T^2)
$$

for attention weights, ignoring batch and head dimensions. The score computation scales like:

$$
O(T^2d_k)
$$

This gives a tradeoff:

| Property | Transformer self-attention |
| --- | --- |
| token parallelism | high within a layer |
| path length between tokens | short |
| memory with sequence length | quadratic for dense attention |
| recurrence over positions | removed |

The paper's efficiency claim should be read under the sequence lengths and hardware of the translation experiments. Modern long-context systems need extra engineering or architectural changes because $T^2$ becomes expensive.

## Experiments and Evidence

The main experiments evaluate machine translation on WMT 2014 English-German and English-French. The paper reports strong BLEU scores and reduced training cost compared with the listed baselines. It also includes parsing experiments to show the model is not limited only to translation.

The compact evidence table is:

| Claim | Evidence | What it supports |
| --- | --- | --- |
| Attention-only model can be competitive or better | WMT translation results | sequence transduction under reported protocol |
| Architecture trains efficiently | reported training time and parallel structure | efficiency in the tested implementation/hardware setting |
| Multi-head attention matters | ablation variants over heads and dimensions | component utility within Transformer family |
| Positional encoding is needed but variant can vary | sinusoidal and learned positional encoding comparison | order signal matters; exact encoding can change |
| Some transfer beyond translation | constituency parsing experiment | architecture may generalize beyond translation |

The evidence is strong for the paper's historical claim: recurrence and convolution are not necessary to build a high-performing sequence transduction model in these settings.

It is weaker for broad modern claims like “Transformers solve language,” “attention explains reasoning,” or “attention is always better than recurrence.” Those require later evidence.

## What the Paper Does Not Prove

This paper does not prove that attention is the only useful operation for all sequence problems. It does not prove that dense attention is efficient for every context length. It does not prove that attention weights are faithful explanations. It does not test modern instruction tuning, retrieval-augmented generation, tool use, multimodal learning, protein language modeling, or structure-based molecular modeling.

The paper's claim should be narrowed to:

$$
\text{attention-only encoder-decoder architecture}
\rightarrow
\text{strong translation performance and parallel training under tested settings}
$$

This narrower claim is still enormous. It opened the path to a new default architecture family.

## Common Misreadings

Because the paper became historically important, it is often read backward from today's LLM landscape. That creates several common misreadings.

| Misreading | Better reading |
| --- | --- |
| The paper introduced modern LLMs directly | it introduced the Transformer architecture in a sequence transduction setting |
| Attention is always better than recurrence | attention was better under the reported translation setup and compute tradeoff |
| Attention weights explain model decisions | attention weights are intermediate mixing coefficients, not causal explanations by default |
| Positional encoding solved length extrapolation | sinusoidal encoding gives a defined signal beyond training length, but extrapolation still needs evidence |
| The original block is the modern standard block | later models changed normalization, activation, position encoding, objective, scale, and data |

These distinctions matter for a wiki. A paper note should preserve what the paper actually showed, while concept notes can explain later generalizations.

## How To Read The Architecture Today

When reading the paper today, separate four layers:

| Layer | In the paper | Later expansion |
| --- | --- | --- |
| Architecture | encoder-decoder Transformer | encoder-only, decoder-only, multimodal, graph, protein, vision variants |
| Objective | supervised translation likelihood | language modeling, masked modeling, instruction tuning, RL-style objectives |
| Data | WMT translation and parsing datasets | web-scale corpora, code, multimodal data, scientific sequences |
| Systems | efficient parallel training on reported hardware | distributed training, KV cache, tensor parallelism, long-context kernels |

Most modern Transformer systems combine this paper's architecture family with a different objective and data regime. For example, decoder-only language models keep causal self-attention but drop the encoder-decoder structure. Protein language models may keep token self-attention but change the vocabulary, dataset split, and evaluation claim. Vision Transformers keep the attention block but change tokens into image patches.

The reusable concept is not “translation model.” The reusable concept is:

$$
\text{tokens}
\rightarrow
\text{attention-based mixing}
\rightarrow
\text{position-wise transformation}
\rightarrow
\text{stacked representation}
$$

## Implementation Reading Checklist

If using this paper as an implementation reference, check these details before assuming two Transformer implementations are equivalent.

| Detail | Why it matters |
| --- | --- |
| normalization placement | original post-norm differs from many pre-norm modern models |
| mask convention | causal, padding, and cross-attention masks must be applied on the correct axis |
| embedding scaling | token embeddings and positional encodings may be scaled or initialized differently |
| label smoothing | affects loss, calibration, and reported BLEU behavior |
| learning-rate schedule | warmup and inverse-square-root decay are part of the training recipe |
| weight sharing | source/target embeddings and output projection choices affect parameterization |
| decoding | beam search and length penalty affect translation scores |

The architecture diagram alone is not enough to reproduce the reported result. The training recipe and evaluation protocol are part of the paper's evidence.

## Evidence Discipline

A useful paper note should keep three statements separate:

| Statement type | Example |
| --- | --- |
| paper claim | Transformer improves reported machine translation quality and parallelism |
| later historical fact | Transformers became the dominant architecture family for LLMs |
| personal inference | this architecture is a good starting point for a new sequence task |

The first statement is supported by the paper. The second is supported by later history and later papers. The third needs a task-specific argument. Mixing these statements makes the paper seem to prove more than it actually tested.

For this wiki, the safe routing is:

- paper-specific evidence stays in [[papers/architectures/attention-is-all-you-need|the compact note]];
- reusable formulas go to [[concepts/architectures/attention|Attention]] and [[concepts/architectures/transformer|Transformer]];
- modern LLM conclusions go to [[papers/llm/index|LLM papers]];
- compute and serving implications go to [[infra/index|Infra]] or [[concepts/systems/index|Systems concepts]].

## Relation To Other Architecture Families

The paper did not make recurrence, convolution, or state-space modeling obsolete as concepts. It changed the default architecture for many sequence tasks by making global token mixing practical and effective.

| Family | Useful bias | Transformer contrast |
| --- | --- | --- |
| RNN | compact sequential state | Transformer exposes all pairwise token interactions within a layer |
| CNN | locality and translation-like sharing | Transformer learns content-dependent mixing rather than fixed local filters |
| GNN | relation-aware message passing over graphs | Transformer can be viewed as dense message passing over tokens |
| SSM/Mamba | efficient long sequence recurrence/scan | often revisits the long-context cost that dense attention struggles with |
| MoE | conditional parameter routing | often combined with Transformer blocks rather than replacing attention |

This comparison helps prevent a second overclaim. The Transformer is not “no inductive bias.” It has different inductive biases: token mixing by learned pairwise similarity, permutation behavior modified by positional encoding, and high parallelism at the cost of dense interaction matrices.

## What To Reuse In Future Paper Notes

When reviewing later Transformer papers, reuse this checklist:

| Question | Why it matters |
| --- | --- |
| What token type is used? | wordpiece, residue, atom, patch, graph node, retrieved chunk, tool event |
| What attention pattern is used? | full, causal, cross, local, sparse, graph-masked |
| What positional or structural signal is used? | order, segment, chain, graph distance, 3D relation |
| What objective trains the model? | translation, language modeling, masked modeling, contrastive, preference |
| What evidence supports the architectural change? | benchmark, ablation, scaling, efficiency, transfer |
| What changed besides the architecture? | data, parameters, compute, schedule, tokenizer, evaluation |

This is why `Attention Is All You Need` works well as the first architecture longform: it teaches the reading pattern for many later papers.

## Limitation Table

| Limitation | Why it matters |
| --- | --- |
| Translation-centered evidence | later LLM behavior is not directly tested |
| BLEU-centered comparison | translation metric does not cover all linguistic or reasoning quality |
| Hardware-dependent efficiency | speed claims depend on implementation and accelerator behavior |
| Dense attention cost | long sequences require later modifications |
| Post-norm original block | many modern implementations differ in stability details |
| No modern pretraining objective | paper predates today's large-scale decoder-only LLM recipe |

## Concept Map

If the review feels too dense, open these notes in order:

1. [[concepts/architectures/attention|Attention]]
2. [[concepts/architectures/softmax|Softmax]]
3. [[concepts/architectures/positional-encoding|Positional encoding]]
4. [[concepts/architectures/transformer|Transformer]]
5. [[concepts/architectures/encoder-decoder|Encoder-decoder architectures]]
6. [[concepts/architectures/computational-complexity|Computational complexity]]

For later descendants, continue to:

- [[papers/llm/index|LLM papers]]
- [[ai/architectures|AI architectures]]
- [[ai/learning-methods|Learning methods]]
- [[concepts/architectures/state-space-model|State-space models]]

## Takeaways

The paper should leave three ideas in memory.

First, attention can be the core sequence-mixing operation, not only an add-on to recurrence.

Second, the Transformer works because it is a whole architecture, not just one attention equation. Multi-head attention, feed-forward layers, residual connections, normalization, positional encoding, masking, optimizer schedule, and regularization all matter.

Third, evidence scope matters. The paper's experiments justify the Transformer as a strong architecture for sequence transduction under the tested translation and parsing settings. The modern LLM story builds on this paper, but also adds scale, data, objectives, decoder-only design, infrastructure, alignment, retrieval, and tool-use workflows.

## Open Questions for Follow-Up

- Which later paper should be the anchor for decoder-only language modeling?
- Which later paper should be the anchor for pre-norm Transformer stabilization?
- Which later paper should be used for long-context attention alternatives?
- Which paper best connects Transformers to protein or molecular sequence modeling?

## Related

- [[papers/architectures/attention-is-all-you-need|Compact paper note]]
- [[papers/architectures/index|Architecture papers]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/encoder-decoder|Encoder-decoder architectures]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/architectures/softmax|Softmax]]
- [[concepts/architectures/feed-forward-network|Feed-forward network]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/normalization-placement|Normalization placement]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[papers/workflows/longform-paper-review-guide|Longform paper review guide]]
