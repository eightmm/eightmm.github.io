---
title: Dropout
tags:
  - papers
  - architectures
  - regularization
  - dropout
---

# Dropout

> Srivastava, Hinton, Krizhevsky, Sutskever, and Salakhutdinov. "Dropout: A Simple Way to Prevent Neural Networks from Overfitting." JMLR, 2014.

Dropout is often introduced as a small training trick, but the paper's architectural importance is larger than that. It changed how large neural networks could be regularized: instead of training one fixed dense network, training samples many thinned subnetworks by randomly removing units and their connections.

The reusable idea is:

$$
\text{train a stochastic family of subnetworks, then use one deterministic network at test time.}
$$

This is why the paper belongs in architecture notes. It is not a new layer family like [[papers/architectures/deep-residual-learning|ResNet]] or [[papers/architectures/attention-is-all-you-need|Transformer]], but it changed the training-time contract of hidden units, dense heads, and later attention/residual blocks.

## Metadata

| Field | Value |
| --- | --- |
| Paper | Dropout: A Simple Way to Prevent Neural Networks from Overfitting |
| Authors | Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov |
| Venue | JMLR 2014 |
| Main contribution | stochastic unit masking for regularization and approximate model averaging |
| Architecture role | training-time architecture block / regularization mechanism |
| Primary source | [JMLR page](https://www.jmlr.org/papers/v15/srivastava14a.html) |

## Question

Large neural networks overfit, but explicitly training and averaging many large networks is expensive.

The paper asks:

$$
\text{Can one train many implicit subnetworks by random masking, then average them cheaply at test time?}
$$

This matters for architecture reading because overparameterized models are not evaluated as bare mathematical functions. They are trained with recipes that shape the effective function class.

## Core Mechanism

For a hidden representation $h \in \mathbb{R}^d$, sample a binary mask:

$$
m_i \sim \mathrm{Bernoulli}(p)
$$

where $p$ is the probability of keeping a unit. The masked activation is:

$$
\tilde{h} = m \odot h
$$

The next layer receives $\tilde{h}$ instead of $h$ during training:

$$
z = W \tilde{h} + b
$$

Each mini-batch therefore trains a different thinned network. A layer with $d$ maskable units has many possible subnetworks:

$$
\#\text{subnetworks} = 2^d
$$

The number is not important because all subnetworks are trained independently. They share parameters, so training is still one stochastic optimization procedure.

## Inference Scaling

The original dropout view uses a deterministic full network at test time with scaled outgoing weights:

$$
W_{\text{test}} = p W
$$

This approximates averaging the predictions of the many thinned networks sampled during training.

Modern frameworks often use inverted dropout during training:

$$
\tilde{h} = \frac{m}{p} \odot h
$$

Then inference can use:

$$
h_{\text{test}} = h
$$

These are different implementation conventions for the same expectation-preserving idea:

$$
\mathbb{E}\left[\frac{m}{p} \odot h\right] = h
$$

When reading papers, check which convention is assumed. A missing factor of $p$ changes activation scale.

## Model Averaging View

The paper frames dropout as an efficient approximation to model combination.

Without dropout, an ensemble of $K$ neural networks would require:

$$
\hat{y} = \frac{1}{K}\sum_{k=1}^{K} f_{\theta_k}(x)
$$

With dropout, subnetworks share weights. Training samples a mask $m$:

$$
\hat{y}_m = f_{\theta, m}(x)
$$

The ideal prediction would average over masks:

$$
\mathbb{E}_{m \sim \mathrm{Bernoulli}(p)}[f_{\theta,m}(x)]
$$

The test-time scaled network is a cheap approximation to this expectation.

This approximation is exact only for simple linear cases. In deep nonlinear networks, it is a practical heuristic. That distinction matters: dropout's empirical value is strong, but the deterministic test network is not a mathematically exact ensemble average in general.

## Co-Adaptation View

The paper also argues that dropout discourages hidden units from relying on specific other hidden units. If a feature is useful only when another exact unit is present, random masking makes that dependency unreliable.

That gives the informal objective:

$$
\text{learn features useful under many contexts, not one brittle co-adapted context.}
$$

This is the intuition behind saying dropout reduces co-adaptation. For paper reading, treat this as an explanatory lens, not as a directly measured quantity unless the experiment actually measures co-adaptation.

## Architecture Contract

Dropout changes the meaning of a layer during training.

| Component | Without dropout | With dropout |
| --- | --- | --- |
| hidden unit | always available | randomly removed during training |
| outgoing connection | always active | inactive when the source unit is dropped |
| training network | fixed graph | sampled subnetwork per example or mini-batch |
| inference network | same graph as training | full graph with scaling convention |
| regularization pressure | implicit through data and optimizer | robustness to missing units and paths |

The practical contract is:

$$
\text{dropout placement}
+ \text{keep probability}
+ \text{scaling convention}
+ \text{inference policy}
$$

These details are part of the architecture recipe.

## Where It Fits

| Context | Role |
| --- | --- |
| [AlexNet](/papers/architectures/alexnet) | regularized large fully connected layers after convolutional feature extraction |
| [Maxout Networks](/papers/architectures/maxout-networks) | maxout units were designed to pair naturally with dropout |
| [Batch Normalization](/papers/architectures/batch-normalization) | later changed how much dropout was needed in some CNN recipes |
| [Transformer](/papers/architectures/attention-is-all-you-need) | uses residual/attention/embedding dropout as part of the training recipe |
| [BERT](/papers/architectures/bert) | uses dropout in Transformer pretraining and fine-tuning recipes |

Dropout is therefore not only a "classic ML regularizer." It appears inside architecture papers because reported gains often depend on a full recipe:

$$
\text{architecture}
+ \text{optimizer}
+ \text{data augmentation}
+ \text{normalization}
+ \text{dropout}
+ \text{compute}
$$

## Evidence Pattern

The paper evaluates dropout across supervised learning settings, including vision, speech, document classification, and biological data.

The strongest reusable evidence is not one benchmark number. It is the repeated pattern:

| Evidence type | What it supports | What it does not prove alone |
| --- | --- | --- |
| lower test error with dropout | dropout can reduce overfitting | dropout is optimal for every architecture |
| multiple domains | mechanism is not tied to one dataset | all domains need the same keep probability |
| pairing with large nets | useful for high-capacity models | small-data scientific models are automatically solved |
| Maxout interaction | flexible units can benefit from dropout | maxout is always better than ReLU |

When reading old architecture papers, dropout often appears as part of the training recipe rather than the headline contribution. It should still be recorded because it can change the comparison.

## Reading Checklist

Ask these questions when a paper uses dropout:

| Question | Why it matters |
| --- | --- |
| Which activations are dropped? | input, feature, attention, residual, classifier, node, edge, or token dropout are different mechanisms |
| What is the keep/drop probability? | too much masking can underfit; too little may not regularize |
| Is inverted dropout used? | affects training/inference scaling |
| Is dropout disabled at evaluation? | evaluation should usually be deterministic unless MC dropout is part of the method |
| Is dropout isolated in ablations? | architecture gains can be mixed with regularization gains |
| Does normalization change the need for dropout? | BatchNorm, LayerNorm, residual scale, and dropout interact |
| Is the dataset small or noisy? | dropout may help, but leakage, label noise, and split quality often dominate |

## Failure Modes

| Failure mode | Symptom |
| --- | --- |
| overusing dropout | training loss remains high and model underfits |
| wrong inference scaling | train/eval behavior shifts unexpectedly |
| dropout inside recurrence without care | unstable sequence modeling or memory disruption |
| treating dropout as architecture evidence | reported improvement may be regularization, not a new block |
| forgetting evaluation mode | stochastic predictions at final evaluation |
| applying same rate everywhere | attention maps, residual streams, and classifier heads need different assumptions |

## Modern Interpretation

In modern deep learning, dropout is one member of a broader family:

$$
\text{stochastic regularization}
=
\text{feature masking}
+ \text{path dropping}
+ \text{token dropping}
+ \text{noise injection}
$$

Examples include:

| Variant | Masked object |
| --- | --- |
| feature dropout | hidden activations |
| attention dropout | attention probabilities or attention links |
| residual dropout | residual branch output |
| stochastic depth | whole residual branch |
| token dropout | input tokens or patches |
| edge/node dropout | graph structure or node features |

The original paper is still worth reading because it gives the baseline logic: stochastic thinning during training plus deterministic approximation at inference.

## Notes For Architecture Shelf

Dropout should be read before or alongside:

| Paper | Reason |
| --- | --- |
| [AlexNet](/papers/architectures/alexnet) | shows dropout as part of the early deep vision recipe |
| [Maxout Networks](/papers/architectures/maxout-networks) | explicitly designs units to work well with dropout |
| [Batch Normalization](/papers/architectures/batch-normalization) | changes optimization and regularization behavior in networks where dropout was common |
| [ResNet](/papers/architectures/deep-residual-learning) | later deep architectures often shift from dense-head dropout to residual/normalization/augmentation recipes |
| [Transformer](/papers/architectures/attention-is-all-you-need) | dropout appears in residual and attention pathways |

The main takeaway:

$$
\text{Dropout is a training-time architecture modifier, not just a scalar hyperparameter.}
$$

## Related

- [[concepts/architectures/dropout|Dropout]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[papers/architectures/maxout-networks|Maxout Networks]]
- [[papers/architectures/alexnet|AlexNet]]
- [[papers/architectures/batch-normalization|Batch Normalization]]
- [[papers/architectures/attention-is-all-you-need|Attention Is All You Need]]
