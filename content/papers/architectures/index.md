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

## Reading Routes

| Route | Start With | Then Read |
| --- | --- | --- |
| Transformer backbone route | [Attention Is All You Need](/papers/architectures/attention-is-all-you-need) | [BERT](/papers/architectures/bert), [GPT-2](/papers/architectures/gpt-2), [T5](/papers/architectures/t5), [LLaMA](/papers/architectures/llama) |
| Efficient attention route | [Transformer-XL](/papers/architectures/transformer-xl) | [Longformer](/papers/architectures/longformer), [BigBird](/papers/architectures/bigbird), [Reformer](/papers/architectures/reformer), [FlashAttention](/papers/architectures/flashattention) |
| CNN vision route | [AlexNet](/papers/architectures/alexnet) | [VGG](/papers/architectures/vgg), [Inception](/papers/architectures/inception), [ResNet](/papers/architectures/deep-residual-learning), [CoAtNet](/papers/architectures/coatnet), [ConvNeXt](/papers/architectures/convnext) |
| Generative image route | [Generative Adversarial Nets](/papers/architectures/generative-adversarial-nets) | [DCGAN](/papers/architectures/dcgan), [Progressive GAN](/papers/architectures/progressive-growing-of-gans), [BigGAN](/papers/architectures/biggan), [StyleGAN2](/papers/architectures/stylegan2), [StyleGAN3](/papers/architectures/stylegan3) |
| Diffusion and latent route | [DDPM](/papers/architectures/ddpm) | [Taming Transformers](/papers/architectures/taming-transformers), [Latent Diffusion Models](/papers/architectures/latent-diffusion-models), [DiT](/papers/architectures/scalable-diffusion-models-with-transformers) |
| Geometric/scientific route | [Neural Message Passing](/papers/architectures/neural-message-passing-for-quantum-chemistry) | [SchNet](/papers/architectures/schnet), [EGNN](/papers/architectures/egnn), [SE(3)-Transformer](/papers/architectures/se3-transformer), [AlphaFold2](/papers/architectures/alphafold2) |

## Shelves

| Shelf | Read For | Anchor Papers |
| --- | --- | --- |
| Sequence and recurrent models | recurrence, gated memory, token mixing, causal convolution, positional encoding, language-model backbone design, prompt-as-interface behavior, long-context memory, efficient attention, state-space sequence modeling | [Long Short-Term Memory](/papers/architectures/long-short-term-memory), [RNN Encoder-Decoder](/papers/architectures/rnn-encoder-decoder), [Bahdanau Attention](/papers/architectures/neural-machine-translation-align-translate), [WaveNet](/papers/architectures/wavenet), [Attention Is All You Need](/papers/architectures/attention-is-all-you-need), [BERT](/papers/architectures/bert), [DeBERTa](/papers/architectures/deberta), [T5](/papers/architectures/t5), [GPT-2](/papers/architectures/gpt-2), [GPT-3](/papers/architectures/gpt-3), [Transformer-XL](/papers/architectures/transformer-xl), [Longformer](/papers/architectures/longformer), [BigBird](/papers/architectures/bigbird), [Reformer](/papers/architectures/reformer), [Linformer](/papers/architectures/linformer), [RoFormer](/papers/architectures/roformer), [ALiBi](/papers/architectures/alibi), [Performer](/papers/architectures/performer), [FlashAttention](/papers/architectures/flashattention), [S4](/papers/architectures/s4), [LLaMA](/papers/architectures/llama), [Layer Normalization](/papers/architectures/layer-normalization), [RMSNorm](/papers/architectures/root-mean-square-layer-normalization), [Mamba](/papers/architectures/mamba) |
| Attention variants and positional mechanisms | alignment, scaled dot-product attention, recurrence memory, sparse local/global/random attention, LSH attention, low-rank attention, disentangled content-position attention, relative/rotary position handling, linear attention bias, linear attention approximations, IO-aware exact attention, key/value head sharing | [Bahdanau Attention](/papers/architectures/neural-machine-translation-align-translate), [Attention Is All You Need](/papers/architectures/attention-is-all-you-need), [Transformer-XL](/papers/architectures/transformer-xl), [DeBERTa](/papers/architectures/deberta), [Longformer](/papers/architectures/longformer), [BigBird](/papers/architectures/bigbird), [Reformer](/papers/architectures/reformer), [Linformer](/papers/architectures/linformer), [RoFormer](/papers/architectures/roformer), [ALiBi](/papers/architectures/alibi), [Performer](/papers/architectures/performer), [FlashAttention](/papers/architectures/flashattention), [GQA](/papers/architectures/gqa) |
| Feed-forward and gating blocks | token-wise channel mixing, token-axis MLP mixing, MetaFormer scaffold, activation choice, gated MLPs, dense and sparse expert blocks | [MetaFormer](/papers/architectures/metaformer), [MLP-Mixer](/papers/architectures/mlp-mixer), [GELU](/papers/architectures/gelu), [GLU Variants Improve Transformer](/papers/architectures/glu-variants-improve-transformer), [Sparsely-Gated MoE](/papers/architectures/sparsely-gated-moe), [GShard](/papers/architectures/gshard), [Switch Transformer](/papers/architectures/switch-transformer), [GLaM](/papers/architectures/glam), [LLaMA](/papers/architectures/llama) |
| State-space sequence models | recurrent state, convolution view, long-range sequence modeling, selective scan lineage | [S4](/papers/architectures/s4), [Mamba](/papers/architectures/mamba) |
| Vision backbones | locality, depth, width, connectivity, cardinality, channel recalibration, dense prediction, efficient mobile blocks, patch tokenization, token/channel MLP mixing, MetaFormer scaffold, convolution-attention hybrid staging, hierarchy, self-supervised ViT representation learning, masked image pre-training, post-ViT ConvNet modernization | [AlexNet](/papers/architectures/alexnet), [VGG](/papers/architectures/vgg), [Inception](/papers/architectures/inception), [Xception](/papers/architectures/xception), [ResNet](/papers/architectures/deep-residual-learning), [ResNeXt](/papers/architectures/resnext), [DenseNet](/papers/architectures/densenet), [MobileNets](/papers/architectures/mobilenets), [MobileNetV2](/papers/architectures/mobilenetv2), [SENet](/papers/architectures/squeeze-and-excitation-networks), [EfficientNet](/papers/architectures/efficientnet), [U-Net](/papers/architectures/u-net), [Vision Transformer](/papers/architectures/vision-transformer), [MLP-Mixer](/papers/architectures/mlp-mixer), [MetaFormer](/papers/architectures/metaformer), [DINO](/papers/architectures/emerging-properties-in-self-supervised-vision-transformers), [MAE](/papers/architectures/masked-autoencoders-are-scalable-vision-learners), [Swin Transformer](/papers/architectures/swin-transformer), [CoAtNet](/papers/architectures/coatnet), [ConvNeXt](/papers/architectures/convnext) |
| Detection, instance segmentation, and set prediction | proposal networks, region heads, one-stage grid/default-box prediction, multi-scale feature pyramids, dense detector imbalance, mask heads, promptable segmentation, object queries, bipartite matching, dense prediction without NMS | [YOLO](/papers/architectures/yolo), [SSD](/papers/architectures/ssd), [FPN](/papers/architectures/feature-pyramid-networks), [RetinaNet](/papers/architectures/retinanet), [Faster R-CNN](/papers/architectures/faster-r-cnn), [Mask R-CNN](/papers/architectures/mask-r-cnn), [Segment Anything](/papers/architectures/segment-anything), [DETR](/papers/architectures/detr) |
| Multimodal encoders and VLM bridges | dual encoders, contrastive alignment, prompt-defined classifiers, visual token resampling, query bottlenecks, frozen vision/LLM bridging | [CLIP](/papers/architectures/clip), [Flamingo](/papers/architectures/flamingo), [BLIP-2](/papers/architectures/blip-2) |
| Graphs, sets, point clouds, geometric models, and multimodal arrays | permutation behavior, message passing, sampling, structural bias, expressivity, molecular graph learning, continuous geometry, directional messages, equivariant scalar/vector/tensor features, structure-conditioned decoding, unordered inputs, latent bottlenecks | [Neural Message Passing](/papers/architectures/neural-message-passing-for-quantum-chemistry), [SchNet](/papers/architectures/schnet), [DimeNet](/papers/architectures/dimenet), [PaiNN](/papers/architectures/painn), [NequIP](/papers/architectures/nequip), [GVP](/papers/architectures/geometric-vector-perceptrons), [ProteinMPNN](/papers/architectures/proteinmpnn), [GCN](/papers/architectures/gcn), [GraphSAGE](/papers/architectures/graphsage), [Graph Attention Networks](/papers/architectures/graph-attention-networks), [GIN](/papers/architectures/graph-isomorphism-network), [Graphormer](/papers/architectures/graphormer), [PointNet](/papers/architectures/pointnet), [Tensor Field Networks](/papers/architectures/tensor-field-networks), [E(n) Equivariant GNN](/papers/architectures/egnn), [SE(3)-Transformer](/papers/architectures/se3-transformer), [Deep Sets](/papers/architectures/deep-sets), [Set Transformer](/papers/architectures/set-transformer), [Perceiver IO](/papers/architectures/perceiver-io) |
| Neural fields and 3D representations | coordinate networks, implicit fields, differentiable rendering, view synthesis | [NeRF](/papers/architectures/nerf) |
| Scientific structure models | domain-specialized architectures for 3D scientific objects | [AlphaFold2](/papers/architectures/alphafold2), [AlphaFold3](/papers/architectures/alphafold3) |
| Generative architectures | autoregressive density, discrete latents, learned image tokenizers, denoising, latent variables, adversarial games, convolutional GAN design, progressive GAN training, attention-based GANs, large-scale conditional GANs, style-based synthesis, alias-free synthesis, invertible flows, iterative sampling, latent-space generation, distribution modeling, diffusion backbones | [PixelRNN and PixelCNN](/papers/architectures/pixel-recurrent-neural-networks), [Auto-Encoding Variational Bayes](/papers/architectures/auto-encoding-variational-bayes), [VQ-VAE](/papers/architectures/neural-discrete-representation-learning), [Taming Transformers](/papers/architectures/taming-transformers), [Generative Adversarial Nets](/papers/architectures/generative-adversarial-nets), [DCGAN](/papers/architectures/dcgan), [Progressive GAN](/papers/architectures/progressive-growing-of-gans), [SAGAN](/papers/architectures/self-attention-gans), [BigGAN](/papers/architectures/biggan), [StyleGAN](/papers/architectures/stylegan), [StyleGAN2](/papers/architectures/stylegan2), [StyleGAN3](/papers/architectures/stylegan3), [Real NVP](/papers/architectures/real-nvp), [Glow](/papers/architectures/glow), [DDPM](/papers/architectures/ddpm), [Latent Diffusion Models](/papers/architectures/latent-diffusion-models), [DiT](/papers/architectures/scalable-diffusion-models-with-transformers) |
| Conditional compute | sparse routing, expert capacity, sharded execution, scaling under fixed token compute, sparse expert LLMs | [Sparsely-Gated MoE](/papers/architectures/sparsely-gated-moe), [GShard](/papers/architectures/gshard), [Switch Transformer](/papers/architectures/switch-transformer), [GLaM](/papers/architectures/glam) |
| Adaptation blocks | frozen backbones, low-rank updates, trainable parameter efficiency | [LoRA](/papers/architectures/lora) |
| Training-time architecture blocks | normalization, activation scale, residual stability, residual-stream scale control, small-batch vision normalization | [Batch Normalization](/papers/architectures/batch-normalization), [Group Normalization](/papers/architectures/group-normalization), [Layer Normalization](/papers/architectures/layer-normalization), [RMSNorm](/papers/architectures/root-mean-square-layer-normalization), [GELU](/papers/architectures/gelu) |
| Continuous-depth models | residual dynamics, adaptive compute, ODE solvers as layers | [Neural ODE](/papers/architectures/neural-ode) |

## Current Notes

| Paper | Main Architecture | Why it is here |
| --- | --- | --- |
| [ImageNet Classification with Deep CNNs](/papers/architectures/alexnet) | AlexNet | large-scale deep CNN vision milestone; full note started |
| [Very Deep Convolutional Networks](/papers/architectures/vgg) | VGG | depth and small-filter CNN design; full note started |
| [Going Deeper with Convolutions](/papers/architectures/inception) | Inception | multi-branch compute-aware CNN module; full note started |
| [Xception](/papers/architectures/xception) | Xception | depthwise separable convolution as an extreme Inception-style factorization; seed note started |
| [ResNeXt](/papers/architectures/resnext) | ResNeXt | cardinality as a CNN scaling axis through aggregated residual transformations; seed note started |
| [Densely Connected Convolutional Networks](/papers/architectures/densenet) | DenseNet | dense skip connectivity and feature reuse; full note started |
| [MobileNets](/papers/architectures/mobilenets) | MobileNetV1 | depthwise separable convolution and width/resolution multipliers for mobile CNNs; seed note started |
| [MobileNetV2](/papers/architectures/mobilenetv2) | inverted residual CNN | efficient mobile CNN block with depthwise convolution, expansion, and linear bottleneck; full note started |
| [Squeeze-and-Excitation Networks](/papers/architectures/squeeze-and-excitation-networks) | SENet | channel-wise squeeze-and-excitation recalibration block for CNNs; seed note started |
| [EfficientNet](/papers/architectures/efficientnet) | EfficientNet | compound scaling for CNN families; full note started |
| [Attention Is All You Need](/papers/architectures/attention-is-all-you-need) | Transformer | attention-only sequence transduction backbone; full note started |
| [BERT](/papers/architectures/bert) | encoder-only Transformer | bidirectional language representation backbone; full note started |
| [DeBERTa](/papers/architectures/deberta) | disentangled encoder-only Transformer | content-position disentangled attention and enhanced mask decoding for BERT-family encoders; seed note started |
| [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](/papers/architectures/t5) | T5 | encoder-decoder Transformer with unified text-to-text task interface; full note started |
| [Language Models are Unsupervised Multitask Learners](/papers/architectures/gpt-2) | decoder-only Transformer | prompt-conditioned autoregressive LM transfer; full note started |
| [Language Models are Few-Shot Learners](/papers/architectures/gpt-3) | scaled decoder-only Transformer | in-context examples as a task interface for few-shot evaluation without gradient updates; full note started |
| [Transformer-XL](/papers/architectures/transformer-xl) | recurrent Transformer | segment-level recurrence, hidden-state memory, and relative positional attention for long-context language modeling; full note started |
| [Longformer](/papers/architectures/longformer) | sparse long-document Transformer | sliding-window local attention plus task-motivated global attention for long documents; full note started |
| [BigBird](/papers/architectures/bigbird) | block sparse Transformer | local, global, and random attention routes for long sequences with linear sparse attention; full note started |
| [Reformer](/papers/architectures/reformer) | efficient Transformer | LSH attention, reversible residual layers, and chunked feed-forward computation for long sequences; full note started |
| [Linformer](/papers/architectures/linformer) | low-rank attention Transformer | key/value sequence-axis projection based on a low-rank attention assumption; seed note started |
| [RoFormer](/papers/architectures/roformer) | Rotary Position Embedding | query/key rotations that make attention scores relative-position aware; full note started |
| [ALiBi](/papers/architectures/alibi) | attention with linear biases | fixed head-specific distance bias for train-short test-long length extrapolation; full note started |
| [Rethinking Attention with Performers](/papers/architectures/performer) | Performer | FAVOR+ random-feature approximation for linear-time softmax attention; full note started |
| [FlashAttention](/papers/architectures/flashattention) | IO-aware exact attention | tiled online-softmax attention kernel that avoids materializing the full attention matrix in HBM; full note started |
| [GQA](/papers/architectures/gqa) | grouped-query attention | key/value head sharing between MHA and MQA for lower KV-cache cost during decoding; full note started |
| [LLaMA](/papers/architectures/llama) | modern decoder-only Transformer | open foundation model recipe with RMSNorm, RoPE, SwiGLU, and public-data scaling; full note started |
| [Gaussian Error Linear Units](/papers/architectures/gelu) | GELU activation | smooth probability-weighted activation used in BERT-style Transformer feed-forward blocks; seed note started |
| [GLU Variants Improve Transformer](/papers/architectures/glu-variants-improve-transformer) | gated Transformer FFN | GLU, GEGLU, and SwiGLU variants for token-wise feed-forward sublayers; full note started |
| [WaveNet](/papers/architectures/wavenet) | dilated causal convolutional generator | autoregressive raw audio generation with dilated causal convolutions and gated residual blocks; full note started |
| [Long Short-Term Memory](/papers/architectures/long-short-term-memory) | LSTM | gated recurrent memory; full note started |
| [Learning Phrase Representations using RNN Encoder-Decoder](/papers/architectures/rnn-encoder-decoder) | GRU / encoder-decoder | gated recurrent sequence transduction; full note started |
| [Neural Machine Translation by Jointly Learning to Align and Translate](/papers/architectures/neural-machine-translation-align-translate) | additive attention / alignment | dynamic source memory access for RNN encoder-decoder translation; full note started |
| [Batch Normalization](/papers/architectures/batch-normalization) | BatchNorm | normalization as an architecture component; full note started |
| [Group Normalization](/papers/architectures/group-normalization) | GroupNorm | batch-independent channel-group normalization for small-batch vision models; seed note started |
| [Deep Residual Learning](/papers/architectures/deep-residual-learning) | ResNet | residual learning and deep CNN optimization; full note started |
| [U-Net](/papers/architectures/u-net) | U-Net | encoder-decoder CNN with localization skip paths; full note started |
| [An Image is Worth 16x16 Words](/papers/architectures/vision-transformer) | Vision Transformer | image patches as Transformer tokens; full note started |
| [MLP-Mixer](/papers/architectures/mlp-mixer) | all-MLP vision backbone | token-mixing and channel-mixing MLP blocks for image classification without convolution or attention; full note started |
| [MetaFormer Is Actually What You Need for Vision](/papers/architectures/metaformer) | MetaFormer / PoolFormer | Transformer-style scaffold with replaceable token mixer, tested with simple pooling; full note started |
| [Emerging Properties in Self-Supervised Vision Transformers](/papers/architectures/emerging-properties-in-self-supervised-vision-transformers) | DINO / self-supervised ViT | teacher-student self-distillation with ViTs and emergent semantic attention behavior; seed note started |
| [Masked Autoencoders Are Scalable Vision Learners](/papers/architectures/masked-autoencoders-are-scalable-vision-learners) | masked autoencoder ViT | visible-only ViT encoder plus lightweight decoder for scalable masked image pre-training; seed note started |
| [Swin Transformer](/papers/architectures/swin-transformer) | Swin Transformer | shifted-window hierarchical vision Transformer; full note started |
| [CoAtNet](/papers/architectures/coatnet) | convolution-attention hybrid backbone | staged MBConv and relative-attention blocks for data-efficient and scalable vision backbones; full note started |
| [A ConvNet for the 2020s](/papers/architectures/convnext) | ConvNeXt | modernized pure ConvNet backbone after ViT/Swin design lessons; full note started |
| [Learning Transferable Visual Models From Natural Language Supervision](/papers/architectures/clip) | CLIP | dual-encoder vision-language contrastive architecture; full note started |
| [Flamingo](/papers/architectures/flamingo) | visual language model | Perceiver-style visual resampler and gated cross-attention bridge for few-shot multimodal prompting; seed note started |
| [BLIP-2](/papers/architectures/blip-2) | Q-Former bridge | trainable query bottleneck between frozen image encoders and frozen language models; seed note started |
| [You Only Look Once](/papers/architectures/yolo) | one-stage object detector | full-image grid prediction for real-time object detection; seed note started |
| [SSD: Single Shot MultiBox Detector](/papers/architectures/ssd) | one-stage default-box detector | default boxes and multi-feature-map prediction for single-shot object detection; seed note started |
| [Feature Pyramid Networks](/papers/architectures/feature-pyramid-networks) | multi-scale feature pyramid | top-down and lateral CNN feature pyramid for scale-aware dense prediction; seed note started |
| [Focal Loss for Dense Object Detection](/papers/architectures/retinanet) | RetinaNet | FPN-based dense one-stage detector trained with focal loss for foreground-background imbalance; seed note started |
| [Faster R-CNN](/papers/architectures/faster-r-cnn) | two-stage object detector | Region Proposal Network plus shared CNN features for end-to-end proposal-based detection; seed note started |
| [Mask R-CNN](/papers/architectures/mask-r-cnn) | instance segmentation detector | Faster R-CNN with RoIAlign and a parallel mask head for instance-level segmentation; seed note started |
| [Segment Anything](/papers/architectures/segment-anything) | promptable segmentation model | reusable image encoder, prompt encoder, and mask decoder for zero-shot promptable segmentation; seed note started |
| [End-to-End Object Detection with Transformers](/papers/architectures/detr) | DETR | object detection as set prediction with object queries and bipartite matching; full note started |
| [Layer Normalization](/papers/architectures/layer-normalization) | LayerNorm | batch-independent normalization for sequence models; full note started |
| [Root Mean Square Layer Normalization](/papers/architectures/root-mean-square-layer-normalization) | RMSNorm | feature-wise root-mean-square normalization without mean-centering; full note started |
| [Neural Message Passing for Quantum Chemistry](/papers/architectures/neural-message-passing-for-quantum-chemistry) | MPNN | message, update, and readout framework for molecular graph neural networks; full note started |
| [SchNet](/papers/architectures/schnet) | continuous-filter atomistic network | distance-conditioned convolution for molecular energy and force modeling; full note started |
| [Directional Message Passing for Molecular Graphs](/papers/architectures/dimenet) | DimeNet | directed molecular messages with distance and angular basis functions; full note started |
| [Equivariant Message Passing for Tensorial Molecular Properties](/papers/architectures/painn) | PaiNN | scalar and vector atom features for equivariant molecular property and tensorial prediction; full note started |
| [NequIP](/papers/architectures/nequip) | E(3)-equivariant interatomic potential | typed tensor features and equivariant convolutions for data-efficient energy-force modeling; full note started |
| [Learning from Protein Structure with Geometric Vector Perceptrons](/papers/architectures/geometric-vector-perceptrons) | GVP | scalar/vector protein-structure graph blocks for geometric and relational reasoning; full note started |
| [ProteinMPNN](/papers/architectures/proteinmpnn) | structure-conditioned protein MPNN | residue-level message passing and order-agnostic autoregressive sequence design for fixed-backbone protein design; full note started |
| [Semi-Supervised Classification with GCNs](/papers/architectures/gcn) | GCN | graph message passing for node classification; full note started |
| [Inductive Representation Learning on Large Graphs](/papers/architectures/graphsage) | GraphSAGE | sampled neighborhood aggregation for inductive node representation learning on large graphs; full note started |
| [Graph Attention Networks](/papers/architectures/graph-attention-networks) | GAT | learned attention over graph neighborhoods; full note started |
| [How Powerful are Graph Neural Networks?](/papers/architectures/graph-isomorphism-network) | GIN | 1-WL expressivity, injective multiset aggregation, and graph-level readout; full note started |
| [Do Transformers Really Perform Bad for Graph Representation?](/papers/architectures/graphormer) | Graphormer | graph Transformer with centrality, shortest-path, and edge encodings; full note started |
| [PointNet](/papers/architectures/pointnet) | PointNet | direct point-cloud architecture with shared point-wise MLPs and symmetric pooling; full note started |
| [NeRF](/papers/architectures/nerf) | neural radiance field | continuous coordinate-to-density/color scene representation with differentiable volume rendering; full note started |
| [Tensor Field Networks](/papers/architectures/tensor-field-networks) | TFN | rotation-, translation-, and permutation-equivariant 3D point-cloud layers with typed tensor features and spherical-harmonic filters; full note started |
| [E(n) Equivariant Graph Neural Networks](/papers/architectures/egnn) | EGNN | geometric graph message passing with E(n)-equivariant coordinate updates; full note started |
| [SE(3)-Transformers](/papers/architectures/se3-transformer) | SE(3)-Transformer | roto-translation equivariant attention for 3D point clouds and graphs; full note started |
| [Highly accurate protein structure prediction with AlphaFold](/papers/architectures/alphafold2) | AlphaFold2 | MSA-pair reasoning, Evoformer, invariant point attention, recycling, and confidence for protein structure prediction; full note started |
| [Accurate structure prediction of biomolecular interactions with AlphaFold 3](/papers/architectures/alphafold3) | AlphaFold3 | unified biomolecular complex prediction with atom-level diffusion coordinate generation; full note started |
| [Deep Sets](/papers/architectures/deep-sets) | Deep Sets | permutation-invariant set function architecture; full note started |
| [Set Transformer](/papers/architectures/set-transformer) | Set Transformer | attention-based permutation-invariant set modeling; full note started |
| [Perceiver IO](/papers/architectures/perceiver-io) | Perceiver IO | latent-bottleneck attention for structured inputs and outputs; full note started |
| [Auto-Encoding Variational Bayes](/papers/architectures/auto-encoding-variational-bayes) | VAE | amortized inference, reparameterization trick, and ELBO training for neural latent-variable generative models; full note started |
| [Pixel Recurrent Neural Networks](/papers/architectures/pixel-recurrent-neural-networks) | PixelRNN / PixelCNN | tractable autoregressive image density modeling with raster-order pixel conditionals and masked spatial dependencies; seed note started |
| [Neural Discrete Representation Learning](/papers/architectures/neural-discrete-representation-learning) | VQ-VAE | vector-quantized discrete latent codebook plus learned autoregressive prior over codes; seed note started |
| [Taming Transformers for High-Resolution Image Synthesis](/papers/architectures/taming-transformers) | VQGAN + autoregressive Transformer | learned image codebook plus Transformer prior for high-resolution image synthesis; seed note started |
| [Generative Adversarial Nets](/papers/architectures/generative-adversarial-nets) | GAN | implicit generator trained through a discriminator-defined adversarial game; full note started |
| [Unsupervised Representation Learning with DCGANs](/papers/architectures/dcgan) | DCGAN | convolutional generator/discriminator design constraints for stable image GANs; seed note started |
| [Progressive Growing of GANs](/papers/architectures/progressive-growing-of-gans) | Progressive GAN | generator and discriminator topology grows from low to high image resolution during training; seed note started |
| [Self-Attention Generative Adversarial Networks](/papers/architectures/self-attention-gans) | SAGAN | non-local self-attention blocks for long-range spatial dependency modeling in GANs; seed note started |
| [Large Scale GAN Training for High Fidelity Natural Image Synthesis](/papers/architectures/biggan) | BigGAN | class-conditional residual GAN scaling with hierarchical latent injection and truncation sampling; seed note started |
| [A Style-Based Generator Architecture for GANs](/papers/architectures/stylegan) | StyleGAN | mapping network, per-layer style modulation, and stochastic noise injection for high-resolution image synthesis; seed note started |
| [Analyzing and Improving the Image Quality of StyleGAN](/papers/architectures/stylegan2) | StyleGAN2 | weight modulation/demodulation, fixed generator redesign, and path length regularization for artifact reduction; seed note started |
| [Alias-Free Generative Adversarial Networks](/papers/architectures/stylegan3) | StyleGAN3 | alias-free generator signal processing for subpixel translation/rotation equivariance and reduced coordinate locking; seed note started |
| [Density estimation using Real NVP](/papers/architectures/real-nvp) | Real NVP | invertible affine coupling flow with exact likelihood, sampling, and latent inference; full note started |
| [Glow](/papers/architectures/glow) | Glow | normalizing flow with actnorm, affine coupling, and learned invertible $1\times1$ channel mixing; seed note started |
| [Denoising Diffusion Probabilistic Models](/papers/architectures/ddpm) | DDPM | iterative denoising architecture and objective for diffusion generative models; full note started |
| [High-Resolution Image Synthesis with Latent Diffusion Models](/papers/architectures/latent-diffusion-models) | Latent Diffusion Model | autoencoder latent-space diffusion with cross-attention conditioning for efficient high-resolution generation; full note started |
| [Scalable Diffusion Models with Transformers](/papers/architectures/scalable-diffusion-models-with-transformers) | DiT | Transformer denoising backbone over latent image patches for scalable diffusion; seed note started |
| [Neural Ordinary Differential Equations](/papers/architectures/neural-ode) | Neural ODE | continuous-depth residual dynamics; full note started |
| [Efficiently Modeling Long Sequences with Structured State Spaces](/papers/architectures/s4) | S4 | structured state-space sequence layer for practical long-range modeling; full note started |
| [Mamba](/papers/architectures/mamba) | selective SSM | input-dependent state-space sequence modeling; full note started |
| [Outrageously Large Neural Networks](/papers/architectures/sparsely-gated-moe) | sparsely-gated MoE | sparse expert routing that decouples total capacity from active per-token computation; full note started |
| [GShard](/papers/architectures/gshard) | sharded MoE Transformer | conditional-compute Transformer scaling with automatic sharding and top-2 expert routing; full note started |
| [Switch Transformer](/papers/architectures/switch-transformer) | sparse MoE Transformer | top-1 expert routing and conditional compute; full note started |
| [GLaM](/papers/architectures/glam) | sparse expert language model | top-2 MoE scaling route for large autoregressive language models; full note started |
| [LoRA](/papers/architectures/lora) | low-rank adaptation block | frozen pretrained weights with trainable low-rank residual updates for parameter-efficient adaptation; full note started |

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
- [[concepts/architectures/wl-test|Weisfeiler-Lehman Test]]
- [[concepts/architectures/graph-transformer|Graph transformer]]
- [[concepts/modalities/3d-structure|3D structure]]
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
