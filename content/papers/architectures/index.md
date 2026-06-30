---
title: Architecture Papers
tags:
  - papers
  - architectures
  - ai
---

# Architecture Papers

Architecture paper note는 오래 남는 model family, block, inductive bias, scaling route를 다룹니다. 이 페이지는 개별 논문을 모두 펼치는 곳이 아니라, architecture를 읽기 위한 paper shelf입니다.

논문의 오래 남는 기여가 재사용 가능한 architecture나 block이면 이 묶음에 둡니다. 같은 논문이 language-model history에서도 중요하면 노트를 복제하지 말고 [[concepts/llm/index|LLM concepts]]나 [[agents/index|Agents]]로 cross-link합니다.

## Depth Standard

Architecture papers are long-term assets, so a finished note should be more than a short abstract. Short notes are acceptable as first-pass index entries, but canonical architecture papers should eventually reach `full note` depth.

| Depth | Target Length | Use |
| --- | --- | --- |
| Seed note | 5-10 min read | metadata, core claim, equation, method sketch, links |
| Full note | 15-25 min read | architecture walkthrough, equations, diagrams or block tables, evidence, ablations, limits |
| Longform review | 25-40 min read | Korean explanatory post or synthesis, not this paper shelf |

For a full architecture note, include:

- the exact input/output contract and tensor shapes when useful;
- the core equations with symbols defined;
- the architecture block decomposition;
- the benchmark/evidence table and what each result actually supports;
- ablations that separate architecture from data, objective, and compute;
- limitations, failure modes, and later variants;
- links back to reusable concepts rather than long duplicated explanations.

This means the current shelf can contain seed notes while the most important papers are expanded over time.

## Shelves

| Shelf | Read For | Anchor Papers |
| --- | --- | --- |
| Sequence and recurrent models | recurrence, gated memory, token mixing, language-model backbone design | [Long Short-Term Memory](/papers/architectures/long-short-term-memory), [RNN Encoder-Decoder](/papers/architectures/rnn-encoder-decoder), [Bahdanau Attention](/papers/architectures/neural-machine-translation-align-translate), [Attention Is All You Need](/papers/architectures/attention-is-all-you-need), [BERT](/papers/architectures/bert), [T5](/papers/architectures/t5), [GPT-2](/papers/architectures/gpt-2), [LLaMA](/papers/architectures/llama), [Layer Normalization](/papers/architectures/layer-normalization), [Mamba](/papers/architectures/mamba) |
| Vision backbones | locality, depth, width, connectivity, dense prediction, patch tokenization, hierarchy | [AlexNet](/papers/architectures/alexnet), [VGG](/papers/architectures/vgg), [Inception](/papers/architectures/inception), [ResNet](/papers/architectures/deep-residual-learning), [DenseNet](/papers/architectures/densenet), [EfficientNet](/papers/architectures/efficientnet), [U-Net](/papers/architectures/u-net), [Vision Transformer](/papers/architectures/vision-transformer), [Swin Transformer](/papers/architectures/swin-transformer) |
| Detection and set prediction | object queries, bipartite matching, dense prediction without NMS | [DETR](/papers/architectures/detr) |
| Multimodal encoders | dual encoders, contrastive alignment, prompt-defined classifiers | [CLIP](/papers/architectures/clip) |
| Graphs, sets, geometric models, and multimodal arrays | permutation behavior, message passing, structural bias, equivariance, unordered inputs, latent bottlenecks | [GCN](/papers/architectures/gcn), [Graph Attention Networks](/papers/architectures/graph-attention-networks), [Graphormer](/papers/architectures/graphormer), [E(n) Equivariant GNN](/papers/architectures/egnn), [SE(3)-Transformer](/papers/architectures/se3-transformer), [Deep Sets](/papers/architectures/deep-sets), [Set Transformer](/papers/architectures/set-transformer), [Perceiver IO](/papers/architectures/perceiver-io) |
| Scientific structure models | domain-specialized architectures for 3D scientific objects | [AlphaFold2](/papers/architectures/alphafold2), [AlphaFold3](/papers/architectures/alphafold3) |
| Generative architectures | denoising, latent variables, adversarial games, invertible flows, iterative sampling, distribution modeling | [Auto-Encoding Variational Bayes](/papers/architectures/auto-encoding-variational-bayes), [Generative Adversarial Nets](/papers/architectures/generative-adversarial-nets), [Real NVP](/papers/architectures/real-nvp), [DDPM](/papers/architectures/ddpm) |
| Conditional compute | sparse routing, expert capacity, scaling under fixed token compute | [Switch Transformer](/papers/architectures/switch-transformer) |
| Training-time architecture blocks | normalization, activation scale, residual stability | [Batch Normalization](/papers/architectures/batch-normalization), [Layer Normalization](/papers/architectures/layer-normalization) |
| Continuous-depth models | residual dynamics, adaptive compute, ODE solvers as layers | [Neural ODE](/papers/architectures/neural-ode) |

## Current Notes

| Paper | Main Architecture | Why it is here |
| --- | --- | --- |
| [ImageNet Classification with Deep CNNs](/papers/architectures/alexnet) | AlexNet | large-scale deep CNN vision milestone; full note started |
| [Very Deep Convolutional Networks](/papers/architectures/vgg) | VGG | depth and small-filter CNN design; full note started |
| [Going Deeper with Convolutions](/papers/architectures/inception) | Inception | multi-branch compute-aware CNN module; full note started |
| [Densely Connected Convolutional Networks](/papers/architectures/densenet) | DenseNet | dense skip connectivity and feature reuse; full note started |
| [EfficientNet](/papers/architectures/efficientnet) | EfficientNet | compound scaling for CNN families; full note started |
| [Attention Is All You Need](/papers/architectures/attention-is-all-you-need) | Transformer | attention-only sequence transduction backbone; full note started |
| [BERT](/papers/architectures/bert) | encoder-only Transformer | bidirectional language representation backbone; full note started |
| [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](/papers/architectures/t5) | T5 | encoder-decoder Transformer with unified text-to-text task interface; full note started |
| [Language Models are Unsupervised Multitask Learners](/papers/architectures/gpt-2) | decoder-only Transformer | prompt-conditioned autoregressive LM transfer; full note started |
| [LLaMA](/papers/architectures/llama) | modern decoder-only Transformer | open foundation model recipe with RMSNorm, RoPE, SwiGLU, and public-data scaling; full note started |
| [Long Short-Term Memory](/papers/architectures/long-short-term-memory) | LSTM | gated recurrent memory; full note started |
| [Learning Phrase Representations using RNN Encoder-Decoder](/papers/architectures/rnn-encoder-decoder) | GRU / encoder-decoder | gated recurrent sequence transduction; full note started |
| [Neural Machine Translation by Jointly Learning to Align and Translate](/papers/architectures/neural-machine-translation-align-translate) | additive attention / alignment | dynamic source memory access for RNN encoder-decoder translation; full note started |
| [Batch Normalization](/papers/architectures/batch-normalization) | BatchNorm | normalization as an architecture component; full note started |
| [Deep Residual Learning](/papers/architectures/deep-residual-learning) | ResNet | residual learning and deep CNN optimization; full note started |
| [U-Net](/papers/architectures/u-net) | U-Net | encoder-decoder CNN with localization skip paths; full note started |
| [An Image is Worth 16x16 Words](/papers/architectures/vision-transformer) | Vision Transformer | image patches as Transformer tokens; full note started |
| [Swin Transformer](/papers/architectures/swin-transformer) | Swin Transformer | shifted-window hierarchical vision Transformer; full note started |
| [Learning Transferable Visual Models From Natural Language Supervision](/papers/architectures/clip) | CLIP | dual-encoder vision-language contrastive architecture; full note started |
| [End-to-End Object Detection with Transformers](/papers/architectures/detr) | DETR | object detection as set prediction with object queries and bipartite matching; full note started |
| [Layer Normalization](/papers/architectures/layer-normalization) | LayerNorm | batch-independent normalization for sequence models; full note started |
| [Semi-Supervised Classification with GCNs](/papers/architectures/gcn) | GCN | graph message passing for node classification; full note started |
| [Graph Attention Networks](/papers/architectures/graph-attention-networks) | GAT | learned attention over graph neighborhoods; full note started |
| [Do Transformers Really Perform Bad for Graph Representation?](/papers/architectures/graphormer) | Graphormer | graph Transformer with centrality, shortest-path, and edge encodings; full note started |
| [E(n) Equivariant Graph Neural Networks](/papers/architectures/egnn) | EGNN | geometric graph message passing with E(n)-equivariant coordinate updates; full note started |
| [SE(3)-Transformers](/papers/architectures/se3-transformer) | SE(3)-Transformer | roto-translation equivariant attention for 3D point clouds and graphs; full note started |
| [Highly accurate protein structure prediction with AlphaFold](/papers/architectures/alphafold2) | AlphaFold2 | MSA-pair reasoning, Evoformer, invariant point attention, recycling, and confidence for protein structure prediction; full note started |
| [Accurate structure prediction of biomolecular interactions with AlphaFold 3](/papers/architectures/alphafold3) | AlphaFold3 | unified biomolecular complex prediction with atom-level diffusion coordinate generation; full note started |
| [Deep Sets](/papers/architectures/deep-sets) | Deep Sets | permutation-invariant set function architecture; full note started |
| [Set Transformer](/papers/architectures/set-transformer) | Set Transformer | attention-based permutation-invariant set modeling; full note started |
| [Perceiver IO](/papers/architectures/perceiver-io) | Perceiver IO | latent-bottleneck attention for structured inputs and outputs; full note started |
| [Auto-Encoding Variational Bayes](/papers/architectures/auto-encoding-variational-bayes) | VAE | amortized inference, reparameterization trick, and ELBO training for neural latent-variable generative models; full note started |
| [Generative Adversarial Nets](/papers/architectures/generative-adversarial-nets) | GAN | implicit generator trained through a discriminator-defined adversarial game; full note started |
| [Density estimation using Real NVP](/papers/architectures/real-nvp) | Real NVP | invertible affine coupling flow with exact likelihood, sampling, and latent inference; full note started |
| [Denoising Diffusion Probabilistic Models](/papers/architectures/ddpm) | DDPM | iterative denoising architecture and objective for diffusion generative models; full note started |
| [Neural Ordinary Differential Equations](/papers/architectures/neural-ode) | Neural ODE | continuous-depth residual dynamics; full note started |
| [Mamba](/papers/architectures/mamba) | selective SSM | input-dependent state-space sequence modeling; full note started |
| [Switch Transformer](/papers/architectures/switch-transformer) | sparse MoE Transformer | top-1 expert routing and conditional compute; full note started |

## Reading Axes

- 어떤 input structure를 가정하는가: sequence, image, graph, set, 3D coordinate, multimodal context, agent state?
- 어떤 inductive bias가 바뀌는가: locality, permutation behavior, equivariance, recurrence, memory, sparsity, routing, hierarchy?
- sequence length, graph size, atom count, residue count, token count에 따라 complexity가 어떻게 바뀌는가?
- contribution이 new architecture, block replacement, scaling rule, efficiency trick 중 무엇인가?
- evidence가 accuracy, sample quality, transfer, stability, latency, memory, throughput 중 무엇에 관한 것인가?
- ablation이 objective, data, compute 변화와 architecture 변화를 분리할 만큼 강한가?

## Concepts

- [[ai/architectures|Architectures]]
- [[concepts/architectures/index|Architecture concepts]]
- [[concepts/architectures/architecture-selection|Architecture selection]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/encoder-only-transformer|Encoder-only Transformer]]
- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/lstm|LSTM]]
- [[concepts/architectures/gru|GRU]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/tasks/segmentation|Segmentation]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/graph-transformer|Graph transformer]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNNs]]
- [[concepts/architectures/perceiver|Perceiver]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]
- [[concepts/architectures/state-space-model|State-space model]]
- [[concepts/architectures/mamba|Mamba]]
- [[concepts/architectures/mixture-of-experts|Mixture of experts]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/systems/scaling-claim-contract|Scaling claim contract]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]

## Evaluation Risks

- Architecture gain이 parameter, token, data, training step, tuning budget 증가에서 온 것일 수 있습니다.
- New block이 다른 objective나 augmentation policy와 함께 test되었을 수 있습니다.
- Throughput gain은 hardware, kernel, batch size, sequence length, sparse implementation에 의존할 수 있습니다.
- Graph/geometric architecture는 graph construction, coordinate frame choice, template-derived feature를 통해 information을 leak할 수 있습니다.

## Related

- [[ai/paper-intake|AI paper intake]]
- [[concepts/coverage-matrix|Coverage matrix]]
- [[papers/workflows/paper-triage|Paper triage]]
- [[agents/index|Agents]]
- [[papers/workflows/ai-molecular-math-paper-template|AI-Molecular-Math paper template]]
- [[papers/analysis/ablation-map|Ablation map]]
- [[papers/analysis/paper-comparison-matrix|Paper comparison matrix]]
