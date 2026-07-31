---
title: Siamese Neural Networks for One-shot Image Recognition
tags:
  - papers
  - architectures
  - representation-learning
  - metric-learning
  - shared-weights
  - few-shot-learning
---

# Siamese Neural Networks for One-shot Image Recognition

> **One-line claim:** a pairwise verification network with tied weights can learn a reusable similarity function, allowing new classes to be recognized from a single example without retraining a class-specific classifier.

## Citation

- Authors: Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov
- Year: 2015
- Venue: ICML Deep Learning Workshop
- Paper: [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.toronto.edu/~zemel/documents/oneshot1.pdf)
- Historical predecessor: [Signature Verification using a Siamese Time Delay Neural Network](https://papers.nips.cc/paper/1993/hash/288cc0ff022877bd3df94bc9360b9c5d-Abstract.html)

## Why this paper belongs in Architecture Papers

The reusable contribution is not a particular convolutional backbone. It is the **twin-network contract**:

$$
x_1
\xrightarrow{f_\theta}
h_1,
\qquad
x_2
\xrightarrow{f_\theta}
h_2,
\qquad
(h_1,h_2)
\xrightarrow{\text{energy}}
\hat y.
$$

The same function $f_\theta$ processes both inputs. The output is a relation between examples rather than a class logit tied to the identities observed during training.

This architecture is a direct predecessor to [[papers/architectures/facenet|FaceNet]], contrastive representation learning, dual encoders, retrieval models, and many pairwise biological modeling systems. The important distinction is:

$$
\text{learn a relation}
\ne
\text{learn a fixed class vocabulary}.
$$

If a new class appears after training, a relation model can compare a query to its single reference example. A conventional classifier with a fixed output head cannot do this without changing or adapting the head.

## Problem setup

Let $\mathcal{C}_{\mathrm{train}}$ be the classes available during training and $\mathcal{C}_{\mathrm{test}}$ be novel classes. In one-shot classification, the model receives one labeled example for each novel class:

$$
\mathcal{S}
=
\{(x_c,y_c):c\in\mathcal{C}_{\mathrm{test}}\},
\qquad
|\{x_c:y_c=c\}|=1.
$$

Given a query $x_q$, the model must choose the class whose reference example is most similar:

$$
\hat y_q
=
\arg\max_{c\in\mathcal{C}_{\mathrm{test}}}
\operatorname{sim}(x_q,x_c).
$$

The model is trained on pair labels:

$$
y_{ij}
=
\begin{cases}
1 & \text{if }x_i\text{ and }x_j\text{ belong to the same class},\\
0 & \text{otherwise}.
\end{cases}
$$

The training classes and evaluation classes can differ. This split is central to the one-shot claim. Randomly splitting individual examples while keeping every class in both partitions tests interpolation within known classes, not recognition of novel classes.

## Architecture contract

| Component | Input | Output | Role |
| --- | --- | --- | --- |
| Twin encoder 1 | $x_1$ | $h_1$ | maps the first example into feature space |
| Twin encoder 2 | $x_2$ | $h_2$ | applies the same mapping to the second example |
| Weight-tying rule | both branches | shared $\theta$ | enforces a common representation space |
| Distance or energy layer | $h_1,h_2$ | relation score | measures similarity or dissimilarity |
| Binary objective | score and pair label | scalar loss | trains verification rather than class identity |
| One-shot decision rule | query and support examples | predicted class | compares a query to novel references |

The two branches may be written separately for exposition, but they are one parameterized encoder evaluated twice:

$$
h_1=f_\theta(x_1),
\qquad
h_2=f_\theta(x_2).
$$

The parameter sharing is the architectural invariant. Duplicating the branch with independent $\theta_1$ and $\theta_2$ changes the problem because the two outputs no longer have a guaranteed common coordinate system.

## Twin networks and symmetry

A pairwise model should usually be symmetric:

$$
g(f_\theta(x_1),f_\theta(x_2))
=
g(f_\theta(x_2),f_\theta(x_1)).
$$

A symmetric distance such as:

$$
d(h_1,h_2)
=
\|h_1-h_2\|_2
$$

already has this property. A learned relation layer must preserve symmetry explicitly or the training data must include both pair orders.

The original paper uses a weighted $L_1$ distance followed by a sigmoid relation score. Let:

$$
\delta
=
|h_1-h_2|,
$$

where the absolute value is elementwise. A learned weighted distance can be written as:

$$
d_w(h_1,h_2)
=
w^\top\delta+b.
$$

The probability that the pair belongs to the same class is:

$$
\hat p(y=1\mid x_1,x_2)
=
\sigma(d_w(h_1,h_2)),
$$

with the sign convention chosen so that the score increases for the positive relation. In an implementation, whether a larger score means “same” or “different” must be stated because distance and similarity conventions are easy to invert.

## Pair verification objective

With pair label $y\in\{0,1\}$ and predicted probability $\hat p$, binary cross-entropy is:

$$
\mathcal{L}_{\mathrm{pair}}
=
-y\log\hat p
-(1-y)\log(1-\hat p).
$$

For a batch of $B$ pairs:

$$
\mathcal{L}
=
\frac{1}{B}
\sum_{i=1}^{B}
\left[
-y_i\log\hat p_i
-(1-y_i)\log(1-\hat p_i)
\right].
$$

The objective teaches the network to answer a verification question:

> Do these two examples belong to the same class under the training definition?

It does not directly teach a global ordering of all classes, and it does not require the output layer to have one unit per identity. The one-shot classifier is constructed afterward by applying the learned verifier to query-support pairs.

## Contrastive energy interpretation

A more general contrastive energy assigns low energy to positive pairs and high energy to negative pairs:

$$
\mathcal{L}_{\mathrm{contrastive}}
=
y\,d(h_1,h_2)^2
+
(1-y)
\left[
m-d(h_1,h_2)
\right]_+^2,
$$

where $m$ is a margin. This form directly shapes an embedding geometry. The pairwise sigmoid model instead learns a calibrated binary relation score.

The two views are related but not identical:

| Objective | Primary output | Useful interpretation |
| --- | --- | --- |
| Binary pair classification | probability of same/different | verification score |
| Contrastive energy | distance with positive/negative margin | embedding geometry |
| Triplet loss | relative ordering of positive and negative distances | ranking constraint |
| InfoNCE-style loss | positive among in-batch candidates | contrastive retrieval geometry |

The architecture can remain the same while the objective changes. This is why it is useful to separate architecture from learning method in the wiki.

## Convolutional twin encoder

For image inputs, the paper uses convolutional Siamese networks. A branch can be abstracted as:

$$
h
=
f_\theta(x)
=
f_{\theta_L}
\circ\cdots\circ
f_{\theta_2}
\circ f_{\theta_1}(x).
$$

The convolutional layers supply local receptive fields and translation sharing. The Siamese wrapper adds a second axis of sharing:

1. parameters are shared across spatial locations within a convolution;
2. the complete encoder is shared across the two paired inputs.

These are separate inductive biases:

| Sharing level | What it assumes |
| --- | --- |
| convolutional kernel sharing | local patterns can recur across image positions |
| twin branch sharing | the same feature map should be applied to both examples |
| pair objective | class relation can be inferred from the two representations |

Replacing the CNN with a Transformer, graph encoder, or protein encoder preserves the twin-network pattern while changing the input-specific inductive bias.

## One-shot inference

After training, the verifier is applied to each query-support pair:

$$
s_c
=
\hat p(y=1\mid x_q,x_c).
$$

The predicted class is:

$$
\hat y_q
=
\arg\max_{c}
s_c.
$$

For a $K$-way one-shot task, the model performs $K$ pair evaluations. The support examples are not used to update $\theta$:

$$
\theta_{\mathrm{test}}
=
\theta_{\mathrm{train}}.
$$

This no-retraining property is the operational meaning of the paper's transfer claim. It should be distinguished from fine-tuning on the support set, which defines a different adaptation protocol.

## Verification versus classification

The model is trained for verification but evaluated through classification. That bridge requires an assumption:

$$
\text{good same/different judgment}
\Rightarrow
\text{useful ranking among novel support examples}.
$$

The implication can fail when:

- every support example is a poor representative of its class;
- pair scores are poorly calibrated across classes;
- the novel classes have a different visual distribution;
- the training relation does not match the evaluation relation;
- the support set contains multiple modes but only one example is available.

One-shot classification is therefore not just a property of the encoder. It is a property of the encoder, pair objective, support protocol, and evaluation split together.

## Sampling pairs

The number of possible pairs grows quadratically with the number of examples. Pair construction is therefore part of the effective training architecture.

For a dataset with class counts $n_c$, the number of positive pairs is:

$$
N_{+}
=
\sum_c
\binom{n_c}{2},
$$

and the number of all pairs is:

$$
N_{\mathrm{all}}
=
\binom{\sum_c n_c}{2}.
$$

Negative pairs usually dominate. Uniformly sampling all pairs can produce severe imbalance and many uninformative negatives. A sampler may instead:

- balance positive and negative pairs;
- sample across different classes uniformly;
- construct episodes that mimic the one-shot evaluation;
- mine confusing negatives using current embeddings;
- control the number of examples from each training class.

The sampling policy changes which relations the model sees and should be reported alongside the loss.

## Episodic interpretation

An episode can be described by a support set $S$ and query set $Q$. For one-shot classification:

$$
S
=
\{(x_1,y_1),\ldots,(x_K,y_K)\},
$$

with one support example per class. The verifier scores each query against all support examples:

$$
\hat y_q
=
\arg\max_{(x_c,y_c)\in S}
\hat p(y=1\mid x_q,x_c).
$$

Training on pair labels and training on episodic classification are related but not equivalent. Pair training may use broader pair combinations, while episodic training matches the test-time support/query structure more closely.

## What the experiments establish

The paper studies character recognition with Omniglot and cross-domain evaluation involving MNIST. The reported results support the claim that a convolutional Siamese verifier can learn features that transfer to novel alphabets or classes with very few examples.

The narrower architectural lesson is:

> A shared encoder plus a pairwise relation function can turn a fixed-class recognition problem into a reusable verification problem.

The results do not establish that one-shot learning is solved generally. Performance depends on the diversity of training classes, pair construction, image preprocessing, support size, and how closely the novel classes match the assumptions learned during training.

## Ablation questions

- Does pairwise verification transfer better than a classifier trained on the same encoder and data?
- How much does weight tying matter compared with two independently trained branches?
- Does the choice of distance or energy function change cross-class transfer?
- Does episodic training outperform random pair training under the same number of examples?
- How sensitive is one-shot accuracy to the number and diversity of training alphabets?
- Does hard-negative mining help after controlling for pair count and compute?
- Does the verifier score remain calibrated when all candidate classes are novel?
- What happens when the single support example is atypical or corrupted?
- How much of the gain comes from the convolutional backbone rather than the Siamese objective?

These experiments separate the contribution of the branch architecture, pair objective, sampler, and evaluation protocol.

## Relation to FaceNet

[[papers/architectures/facenet|FaceNet]] keeps the shared encoder idea but makes the embedding geometry and triplet loss more explicit:

$$
f_\theta(x)
\rightarrow
z,
\qquad
d(z^a,z^p)+\alpha
\le
d(z^a,z^n).
$$

The progression is:

| Stage | Main contract |
| --- | --- |
| Siamese verification | classify whether two inputs match |
| Contrastive embedding | pull positive pairs together and push negatives apart |
| FaceNet-style triplet embedding | enforce relative distance ordering |
| Modern dual encoder | align two modalities or views with in-batch negatives |

The architecture family remains recognizable even as the loss, sampler, and downstream retrieval system evolve.

## Relation to computational biology

The same architecture can compare biological objects, but “same class” must be defined at the domain level. Possible pair relations include:

- two protein sequences from the same family;
- two molecular conformers of the same compound;
- two ligands with a specified activity relation under the same assay;
- two pocket representations from the same binding-site family;
- two augmented views that should preserve a structural identity.

The relation label is not interchangeable across these cases. A pairwise model can learn assay artifacts, scaffold shortcuts, or protein-family leakage if the split is not designed around the intended deployment question.

For a molecule pair $(m_i,m_j)$, a domain-specific relation might be:

$$
y_{ij}
=
\mathbf{1}
\left[
\operatorname{activity}(m_i,m_j)
\text{ satisfies a predeclared rule}
\right].
$$

The rule, assay context, scaffold split, and evaluation metric must be stored with the model result. The Siamese architecture does not make an ambiguous relation scientifically meaningful.

## Complexity and deployment

For an encoder cost $C_f$ and pair relation cost $C_g$, direct comparison of a query with $K$ support examples costs approximately:

$$
O\left(K(C_f+C_g)\right)
$$

if the support embeddings are recomputed, or:

$$
O\left(C_f+K C_g\right)
$$

if support embeddings are cached.

For a shared encoder with embedding dimension $d$, storing $N$ reference examples requires:

$$
O(Nd)
$$

memory before index overhead. This makes the architecture attractive for open-set retrieval, but the pair relation may still require a full pairwise pass if the energy layer is not decomposable.

Deployment questions include:

- can support embeddings be cached safely?
- does preprocessing match between query and support?
- is the similarity threshold calibrated after quantization?
- does retrieval latency grow linearly with support size?
- should an approximate nearest-neighbor index replace direct pair evaluation?

## Failure modes

### Pair-label ambiguity

The same pair can be considered positive under one relation and negative under another. If the label definition is not tied to the deployment task, the embedding geometry becomes difficult to interpret.

### Shortcut learning

The model may use background, acquisition device, batch, or class-specific artifacts rather than the intended object identity. Cross-domain or scaffold-aware splits are needed to expose this.

### Support sensitivity

One-shot inference is highly sensitive to the sole support example. The model may have learned class prototypes that do not represent rare or multimodal classes.

### Calibration drift

The largest pair score in a candidate set is not necessarily a calibrated probability. Changing the number of candidates can change the maximum score even when the query is unchanged.

### Branch inconsistency

Accidentally using independent parameters, separate normalization statistics, or different preprocessing in the two branches breaks the shared-space assumption.

## Reproduction checklist

- [ ] State whether the two branches share every encoder parameter and preprocessing step.
- [ ] Define positive and negative pair semantics before sampling.
- [ ] Report pair balance, class balance, and hard-negative policy.
- [ ] Separate training classes from novel evaluation classes.
- [ ] Specify whether evaluation uses pair verification or one-shot ranking.
- [ ] Record support size, episode construction, and query selection.
- [ ] Report the energy or distance function and its score direction.
- [ ] Compare against a fixed-class classifier with matched backbone and compute.
- [ ] Test sensitivity to atypical support examples and candidate-set size.
- [ ] For biological data, report assay context, scaffold/family split, and relation-label provenance.

## Takeaway

Siamese Networks introduced a durable architecture pattern:

$$
\boxed{
\text{shared encoder}
\rightarrow
\text{common representation space}
\rightarrow
\text{pairwise relation}
}
$$

The pattern changes what can be learned and deployed. Instead of memorizing a fixed output vocabulary, the model learns a reusable comparison rule. The key engineering lesson is that weight tying, pair construction, relation semantics, and evaluation split form one contract. Changing any one of them changes the meaning of the result.

## Related notes

- [[ai/architectures|Architectures]]
- [[papers/architectures/facenet|FaceNet]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/architectures/embedding|Embedding]]
- [[papers/architectures/clip|CLIP]]
- [[molecular-modeling/index|Computational Biology]]
