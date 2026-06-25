---
title: Concepts
tags:
  - concepts
---

# Concepts

Concept notes define reusable terms used across research notes, paper summaries, project pages, and infrastructure notes.

Use this page as a map. Start from a hub when exploring a field, then move into individual notes when a paper, project, or post needs a precise definition.

## Main Hubs

- [[concepts/math/index|Math foundations]]
- [[concepts/machine-learning/index|Machine learning]]
- [[concepts/data/index|Data]]
- [[concepts/modalities/index|Modalities]]
- [[concepts/tasks/index|Tasks]]
- [[concepts/architectures/index|Architectures]]
- [[concepts/learning/index|Learning methods]]
- [[concepts/generative-models/index|Generative models]]
- [[concepts/systems/index|AI systems]]
- [[concepts/research-methodology/index|Research methodology]]
- [[concepts/llm/index|LLM concepts]]
- [[concepts/evaluation/index|Evaluation]]
- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/molecular-modeling/index|Molecular modeling concepts]]
- [[concepts/sbdd/index|Structure-based drug discovery]]
- [[concepts/protein-modeling/index|Protein modeling concepts]]
- [[concepts/genome-modeling/index|Genome modeling concepts]]
- [[entities/index|Entities]]

## Reading Paths

- AI basics: [[concepts/math/index|Math foundations]] -> [[concepts/machine-learning/empirical-risk-minimization|Empirical risk minimization]] -> [[concepts/machine-learning/stochastic-gradient|Stochastic gradient]] -> [[concepts/data/index|Data]] -> [[concepts/architectures/index|Architectures]] -> [[concepts/learning/index|Learning methods]] -> [[concepts/learning/augmentation-policy|Augmentation policy]] -> [[concepts/evaluation/evaluation-protocol|Evaluation protocol]] -> [[concepts/evaluation/index|Evaluation]]
- Math path: [[concepts/math/linear-algebra|Linear algebra]] -> [[concepts/math/calculus|Calculus]] -> [[concepts/math/matrix-calculus|Matrix calculus]] -> [[concepts/math/probability-distribution|Probability distribution]] -> [[concepts/math/expectation|Expectation]] -> [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]] -> [[concepts/math/maximum-likelihood|Maximum likelihood]] -> [[concepts/math/entropy-kl|Entropy and KL divergence]]
- Data path: [[entities/dataset|Dataset]] -> [[concepts/data/dataset-construction-checklist|Dataset construction checklist]] -> [[concepts/data/example-unit|Example unit]] -> [[concepts/data/split-unit|Split unit]] -> [[concepts/data/preprocessing-contract|Preprocessing contract]] -> [[concepts/data/label-semantics|Label semantics]] -> [[concepts/data/dataset-shift|Dataset shift]] -> [[concepts/data/benchmark|Benchmark]]
- Modality path: [[concepts/modalities/text|Text]] -> [[concepts/modalities/sequence|Sequence]] -> [[concepts/modalities/image|Image]] -> [[concepts/modalities/video|Video]] -> [[concepts/modalities/audio|Audio]] -> [[concepts/modalities/tabular|Tabular]] -> [[concepts/modalities/graph|Graph]] -> [[concepts/modalities/3d-structure|3D structure]] -> [[concepts/modalities/modality-alignment|Modality alignment]] -> [[concepts/modalities/missing-modality|Missing modality]] -> [[concepts/modalities/multimodal-learning|Multimodal learning]]
- Task path: [[concepts/tasks/task-specification|Task specification]] -> [[concepts/tasks/retrieval|Retrieval]] -> [[concepts/tasks/question-answering|Question answering]] -> [[concepts/tasks/sequence-generation|Sequence generation]] -> [[concepts/tasks/structured-prediction|Structured prediction]]
- Deep learning architecture: [[concepts/architectures/inductive-bias|Inductive bias]] -> [[concepts/architectures/parameter-sharing|Parameter sharing]] -> [[concepts/architectures/computational-complexity|Computational complexity]] -> [[concepts/architectures/tokenization|Tokenization]] -> [[concepts/architectures/embedding|Embedding]] -> [[concepts/architectures/attention|Attention]] -> [[concepts/architectures/transformer|Transformer]]
- Systems path: [[concepts/systems/training-run|Training run]] -> [[concepts/systems/resource-scheduling|Resource scheduling]] -> [[concepts/systems/checkpoint-state|Checkpoint state]] -> [[concepts/systems/inference|Inference]] -> [[concepts/systems/batch-online-inference|Batch and online inference]] -> [[concepts/systems/model-serving|Model serving]] -> [[concepts/systems/failure-recovery|Failure recovery]] -> [[concepts/systems/reproducibility|Reproducibility]]
- Research method path: [[concepts/research-methodology/research-question|Research question]] -> [[concepts/research-methodology/hypothesis|Hypothesis]] -> [[concepts/research-methodology/minimum-viable-experiment|Minimum viable experiment]] -> [[concepts/research-methodology/threat-to-validity|Threat to validity]] -> [[concepts/research-methodology/result-interpretation|Result interpretation]]
- Agent path: [[agents/core/agent-architecture|Agent architecture]] -> [[agents/core/agent-environment|Agent environment]] -> [[agents/core/action-space|Action space]] -> [[agents/tools/tool-contract|Tool contract]] -> [[agents/tools/tool-result-handling|Tool result handling]] -> [[agents/verification/acceptance-criteria|Acceptance criteria]] -> [[agents/verification/verification-loop|Verification loop]]
- Graph and geometry: [[concepts/math/geometry|Geometry]] -> [[concepts/math/symmetry-group|Symmetry group]] -> [[concepts/geometric-deep-learning/coordinate-frame|Coordinate frame]] -> [[concepts/geometric-deep-learning/distance-geometry|Distance geometry]] -> [[concepts/architectures/graph-construction|Graph construction]] -> [[concepts/architectures/gnn|Graph neural networks]] -> [[concepts/geometric-deep-learning/invariant-feature|Invariant feature]] -> [[concepts/geometric-deep-learning/equivariant-feature|Equivariant feature]] -> [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]
- Structure-based AI: [[entities/pocket|Pocket]] -> [[concepts/protein-modeling/binding-site|Binding site]] -> [[concepts/protein-modeling/pocket-representation|Pocket representation]] -> [[concepts/sbdd/receptor-ligand-preparation|Receptor and ligand preparation]] -> [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]] -> [[concepts/sbdd/docking-workflow|Docking workflow]] -> [[concepts/sbdd/pose-generation|Pose generation]] -> [[concepts/sbdd/pose-quality|Pose quality]] -> [[concepts/sbdd/scoring-function|Scoring function]] -> [[concepts/sbdd/binding-affinity|Binding affinity]] -> [[concepts/sbdd/virtual-screening|Virtual screening]]
- Molecular modeling: [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]] -> [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]] -> [[concepts/molecular-modeling/smiles|SMILES]] -> [[concepts/molecular-modeling/molecular-graph|Molecular graph]] -> [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]] -> [[concepts/molecular-modeling/molecular-similarity|Molecular similarity]] -> [[concepts/molecular-modeling/molecular-property-prediction|Molecular property prediction]] -> [[concepts/molecular-modeling/conformer|Conformer]]
- Protein modeling: [[entities/sequence|Sequence]] -> [[concepts/protein-modeling/protein-domain|Protein domain]] -> [[concepts/protein-modeling/protein-representation|Protein representation]] -> [[concepts/protein-modeling/contact-map|Contact map]] -> [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]]
- Genome-level sequence: [[entities/genome|Genome]] -> [[concepts/genome-modeling/genomic-region|Genomic region]] -> [[concepts/genome-modeling/k-mer|K-mer]] -> [[concepts/genome-modeling/variant-effect-prediction|Variant effect prediction]]
- Paper reading: [[papers/paper-triage|Paper triage]] -> [[papers/paper-note-format|Paper note format]] -> [[papers/claim-extraction|Claim extraction]] -> [[papers/benchmark-card|Benchmark card]] -> [[papers/ablation-map|Ablation map]] -> [[papers/reproduction-plan|Reproduction plan]] -> [[papers/reading-status|Reading status]]
- LLM Wiki: [[concepts/llm/language-model|Language model]] -> [[concepts/llm/context-window|Context window]] -> [[concepts/llm/token-budget|Token budget]] -> [[concepts/llm/context-packing|Context packing]] -> [[concepts/llm/embedding-retrieval|Embedding retrieval]] -> [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]] -> [[concepts/llm/prompt-injection-boundary|Prompt injection boundary]]

## Entity Concepts

- [[entities/index|Entities]]
- [[entities/entity-relation-map|Entity relation map]]
- [[entities/target|Target]]
- [[entities/protein|Protein]]
- [[entities/pocket|Pocket]]
- [[entities/ligand|Ligand]]
- [[entities/molecule|Molecule]]
- [[entities/protein-ligand-complex|Protein-ligand complex]]

## Math Foundations

- [[concepts/math/index|Math foundations]]
- [[concepts/math/linear-algebra|Linear algebra]]
- [[concepts/math/eigenvalue-eigenvector|Eigenvalue and eigenvector]]
- [[concepts/math/singular-value-decomposition|Singular value decomposition]]
- [[concepts/math/calculus|Calculus]]
- [[concepts/math/matrix-calculus|Matrix calculus]]
- [[concepts/math/geometry|Geometry]]
- [[concepts/math/symmetry-group|Symmetry group]]
- [[concepts/math/random-variable|Random variable]]
- [[concepts/math/probability-distribution|Probability distribution]]
- [[concepts/math/expectation|Expectation]]
- [[concepts/math/covariance-correlation|Covariance and correlation]]
- [[concepts/math/monte-carlo-estimation|Monte Carlo estimation]]
- [[concepts/math/maximum-likelihood|Maximum likelihood]]
- [[concepts/math/entropy-kl|Entropy and KL divergence]]
- [[concepts/math/bayes-rule|Bayes rule]]
- [[concepts/math/statistical-estimator|Statistical estimator]]
- [[concepts/math/bias-variance-tradeoff|Bias-variance tradeoff]]

## Data

- [[concepts/data/index|Data]]
- [[entities/dataset|Dataset]]
- [[concepts/data/dataset-construction-checklist|Dataset construction checklist]]
- [[concepts/data/dataset-card|Dataset card]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/data/data-distribution|Data distribution]]
- [[concepts/data/data-schema|Data schema]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/data/data-curation|Data curation]]
- [[concepts/data/data-versioning|Data versioning]]
- [[concepts/data/annotation-labeling|Annotation and labeling]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/label-noise|Label noise]]
- [[concepts/data/dataset-shift|Dataset shift]]
- [[concepts/data/sampling-strategy|Sampling strategy]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/data/benchmark|Benchmark]]
- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]

## Modalities

- [[concepts/modalities/index|Modalities]]
- [[concepts/modalities/text|Text]]
- [[concepts/modalities/sequence|Sequence]]
- [[concepts/modalities/image|Image]]
- [[concepts/modalities/video|Video]]
- [[concepts/modalities/audio|Audio]]
- [[concepts/modalities/tabular|Tabular]]
- [[concepts/modalities/graph|Graph]]
- [[concepts/modalities/3d-structure|3D structure]]
- [[concepts/modalities/modality-alignment|Modality alignment]]
- [[concepts/modalities/missing-modality|Missing modality]]
- [[concepts/modalities/multimodal-learning|Multimodal learning]]

## Tasks

- [[concepts/tasks/index|Tasks]]
- [[concepts/tasks/task-specification|Task specification]]
- [[concepts/machine-learning/classification|Classification]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/machine-learning/ranking|Ranking]]
- [[concepts/tasks/retrieval|Retrieval]]
- [[concepts/tasks/object-detection|Object detection]]
- [[concepts/tasks/segmentation|Segmentation]]
- [[concepts/tasks/captioning|Captioning]]
- [[concepts/tasks/question-answering|Question answering]]
- [[concepts/tasks/sequence-generation|Sequence generation]]
- [[concepts/tasks/structured-prediction|Structured prediction]]
- [[concepts/tasks/time-series-forecasting|Time-series forecasting]]
- [[concepts/tasks/anomaly-detection|Anomaly detection]]

## Architectures

- [[concepts/architectures/index|Architectures]]
- [[concepts/architectures/inductive-bias|Inductive bias]]
- [[concepts/architectures/parameter-sharing|Parameter sharing]]
- [[concepts/architectures/architecture-selection|Architecture selection]]
- [[concepts/architectures/computational-complexity|Computational complexity]]
- [[concepts/architectures/linear-layer|Linear layer]]
- [[concepts/architectures/activation-function|Activation function]]
- [[concepts/architectures/normalization|Normalization]]
- [[concepts/architectures/residual-connection|Residual connection]]
- [[concepts/architectures/tokenization|Tokenization]]
- [[concepts/architectures/embedding|Embedding]]
- [[concepts/architectures/positional-encoding|Positional encoding]]
- [[concepts/architectures/attention|Attention]]
- [[concepts/architectures/cross-attention|Cross-attention]]
- [[concepts/architectures/pooling-readout|Pooling and readout]]
- [[concepts/architectures/mlp|MLP]]
- [[concepts/architectures/cnn|CNN]]
- [[concepts/architectures/residual-network|Residual network]]
- [[concepts/architectures/u-net|U-Net]]
- [[concepts/architectures/rnn|RNN]]
- [[concepts/architectures/graph-construction|Graph construction]]
- [[concepts/architectures/gnn|Graph neural networks]]
- [[concepts/architectures/deep-sets|Deep Sets]]
- [[concepts/architectures/transformer|Transformer]]
- [[concepts/architectures/encoder-only-transformer|Encoder-only Transformer]]
- [[concepts/architectures/decoder-only-transformer|Decoder-only Transformer]]
- [[concepts/architectures/vision-transformer|Vision Transformer]]
- [[concepts/architectures/state-space-model|State-space models]]
- [[concepts/architectures/mamba|Mamba]]
- [[concepts/architectures/mixture-of-experts|Mixture of experts]]

## Machine Learning

- [[concepts/machine-learning/index|Machine learning]]
- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]
- [[concepts/machine-learning/feature-engineering|Feature engineering]]
- [[concepts/machine-learning/classification|Classification]]
- [[concepts/machine-learning/regression|Regression]]
- [[concepts/machine-learning/ranking|Ranking]]
- [[concepts/machine-learning/density-estimation|Density estimation]]
- [[concepts/machine-learning/representation-learning|Representation learning]]
- [[concepts/machine-learning/empirical-risk-minimization|Empirical risk minimization]]
- [[concepts/machine-learning/loss-function|Loss function]]
- [[concepts/machine-learning/training-loop|Training loop]]
- [[concepts/machine-learning/stochastic-gradient|Stochastic gradient]]
- [[concepts/machine-learning/gradient-descent|Gradient descent]]
- [[concepts/machine-learning/backpropagation|Backpropagation]]
- [[concepts/machine-learning/optimizer|Optimizer]]
- [[concepts/machine-learning/learning-rate-schedule|Learning rate schedule]]
- [[concepts/machine-learning/weight-decay|Weight decay]]
- [[concepts/machine-learning/gradient-clipping|Gradient clipping]]
- [[concepts/machine-learning/batch-size|Batch size]]
- [[concepts/machine-learning/linear-model|Linear model]]
- [[concepts/machine-learning/tree-based-model|Tree-based model]]
- [[concepts/machine-learning/kernel-method|Kernel method]]
- [[concepts/machine-learning/ensemble-method|Ensemble method]]
- [[concepts/machine-learning/regularization|Regularization]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[concepts/machine-learning/dimensionality-reduction|Dimensionality reduction]]

## Learning Methods

- [[concepts/learning/index|Learning methods]]
- [[concepts/learning/supervised-learning|Supervised learning]]
- [[concepts/learning/pretraining|Pretraining]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/learning/augmentation-policy|Augmentation policy]]
- [[concepts/learning/representation-collapse|Representation collapse]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/learning/transfer-learning|Transfer learning]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/instruction-tuning|Instruction tuning]]
- [[concepts/learning/domain-adaptation|Domain adaptation]]
- [[concepts/learning/curriculum-learning|Curriculum learning]]
- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[concepts/learning/preference-optimization|Preference optimization]]

## Structure-Based AI

- [[concepts/sbdd/index|Structure-based drug discovery]]
- [[concepts/sbdd/receptor-ligand-preparation|Receptor and ligand preparation]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/protein-ligand-interaction|Protein-ligand interaction]]
- [[concepts/sbdd/docking-workflow|Docking workflow]]
- [[concepts/sbdd/scoring-function|Scoring function]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/sbdd/interaction-fingerprint|Interaction fingerprint]]
- [[concepts/sbdd/binding-affinity|Binding affinity]]
- [[concepts/sbdd/virtual-screening|Virtual screening]]
- [[concepts/sbdd/template-leakage|Template leakage]]

## Molecular Modeling

- [[concepts/molecular-modeling/index|Molecular modeling concepts]]
- [[concepts/molecular-modeling/molecular-standardization|Molecular standardization]]
- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/molecular-modeling/smiles|SMILES]]
- [[concepts/molecular-modeling/molecular-graph|Molecular graph]]
- [[concepts/molecular-modeling/molecular-fingerprint|Molecular fingerprint]]
- [[concepts/molecular-modeling/molecular-similarity|Molecular similarity]]
- [[concepts/molecular-modeling/molecular-property-prediction|Molecular property prediction]]
- [[concepts/molecular-modeling/substructure-search|Substructure search]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/molecular-modeling/tautomer|Tautomer]]
- [[concepts/molecular-modeling/protonation-state|Protonation state]]
- [[concepts/molecular-modeling/stereochemistry|Stereochemistry]]

## Protein Modeling

- [[concepts/protein-modeling/index|Protein modeling concepts]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/protein-modeling/protein-domain|Protein domain]]
- [[concepts/protein-modeling/binding-site|Binding site]]
- [[concepts/protein-modeling/pocket-representation|Pocket representation]]
- [[concepts/protein-modeling/protein-structure-prediction|Protein structure prediction]]
- [[concepts/protein-modeling/multiple-sequence-alignment|Multiple sequence alignment]]
- [[concepts/protein-modeling/sequence-identity-clustering|Sequence identity clustering]]
- [[concepts/protein-modeling/contact-map|Contact map]]
- [[concepts/protein-modeling/sequence-structure-alignment|Sequence-structure alignment]]

## Genome Modeling

- [[concepts/genome-modeling/index|Genome modeling concepts]]
- [[concepts/genome-modeling/genomic-region|Genomic region]]
- [[concepts/genome-modeling/k-mer|K-mer]]
- [[concepts/genome-modeling/variant-effect-prediction|Variant effect prediction]]
- [[concepts/genome-modeling/genome-annotation|Genome annotation]]

## Generative Models

- [[concepts/generative-models/index|Generative models]]
- [[concepts/generative-models/latent-variable-model|Latent variable model]]
- [[concepts/generative-models/conditional-generation|Conditional generation]]
- [[concepts/generative-models/sampling|Sampling]]
- [[concepts/generative-models/guidance|Guidance]]
- [[concepts/generative-models/autoregressive-model|Autoregressive model]]
- [[concepts/generative-models/diffusion-model|Diffusion model]]
- [[concepts/generative-models/score-based-model|Score-based model]]
- [[concepts/generative-models/flow-matching|Flow matching]]
- [[concepts/generative-models/rectified-flow|Rectified flow]]
- [[concepts/generative-models/normalizing-flow|Normalizing flow]]
- [[concepts/generative-models/vae|VAE]]
- [[concepts/generative-models/gan|GAN]]
- [[concepts/generative-models/molecular-generation|Molecular generation]]
- [[concepts/generative-models/protein-design|Protein design]]

## AI Systems

- [[concepts/systems/index|AI systems]]
- [[concepts/systems/training-run|Training run]]
- [[concepts/systems/resource-scheduling|Resource scheduling]]
- [[concepts/systems/environment-management|Environment management]]
- [[concepts/systems/inference|Inference]]
- [[concepts/systems/model-serving|Model serving]]
- [[concepts/systems/latency-throughput|Latency and throughput]]
- [[concepts/systems/memory-compute-tradeoff|Memory-compute tradeoff]]
- [[concepts/systems/storage-io|Storage and IO]]
- [[concepts/systems/observability|Observability]]
- [[concepts/systems/experiment-tracking|Experiment tracking]]
- [[concepts/systems/reproducibility|Reproducibility]]

## Research Methodology

- [[concepts/research-methodology/index|Research methodology]]
- [[concepts/research-methodology/research-question|Research question]]
- [[concepts/research-methodology/hypothesis|Hypothesis]]
- [[concepts/research-methodology/experiment-design|Experiment design]]
- [[concepts/research-methodology/minimum-viable-experiment|Minimum viable experiment]]
- [[concepts/research-methodology/result-interpretation|Result interpretation]]
- [[concepts/research-methodology/threat-to-validity|Threat to validity]]
- [[concepts/research-methodology/research-log|Research log]]

## LLM Concepts

- [[concepts/llm/index|LLM concepts]]
- [[concepts/llm/language-model|Language model]]
- [[concepts/llm/context-window|Context window]]
- [[concepts/llm/token-budget|Token budget]]
- [[concepts/llm/context-packing|Context packing]]
- [[concepts/llm/tool-calling|Tool calling]]
- [[concepts/llm/prompt-injection-boundary|Prompt injection boundary]]
- [[concepts/llm/prompting|Prompting]]
- [[concepts/llm/in-context-learning|In-context learning]]
- [[concepts/llm/decoding|Decoding]]
- [[concepts/llm/structured-output|Structured output]]
- [[concepts/llm/hallucination-grounding|Hallucination and grounding]]
- [[concepts/llm/embedding-retrieval|Embedding retrieval]]
- [[concepts/llm/retrieval-augmented-generation|Retrieval-augmented generation]]
- [[agents/core/agent-architecture|Agent architecture]]
- [[agents/workflows/agent-orchestration|Agent orchestration]]

## Geometric Deep Learning

- [[concepts/geometric-deep-learning/index|Geometric deep learning]]
- [[concepts/geometric-deep-learning/geometric-architecture|Geometric architecture]]
- [[concepts/geometric-deep-learning/equivariant-gnn|Equivariant GNN]]

## Learning Methods

- [[concepts/learning/index|Learning methods]]
- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/jepa|JEPA]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/transfer-learning|Transfer learning]]
- [[concepts/learning/preference-optimization|Preference optimization]]

## Evaluation

- [[concepts/evaluation/index|Evaluation]]
- [[concepts/evaluation/metric|Metric]]
- [[concepts/evaluation/confusion-matrix|Confusion matrix]]
- [[concepts/evaluation/threshold-selection|Threshold selection]]
- [[concepts/evaluation/classification-metrics|Classification metrics]]
- [[concepts/evaluation/regression-metrics|Regression metrics]]
- [[concepts/evaluation/ranking-metrics|Ranking metrics]]
- [[concepts/evaluation/generation-evaluation|Generation evaluation]]
- [[concepts/evaluation/baseline|Baseline]]
- [[concepts/evaluation/ablation-study|Ablation study]]
- [[concepts/evaluation/confidence-interval|Confidence interval]]
- [[concepts/evaluation/bootstrap-evaluation|Bootstrap evaluation]]
- [[concepts/evaluation/statistical-significance|Statistical significance]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/cross-validation|Cross-validation]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/evaluation/negative-set|Negative set]]
- [[concepts/evaluation/activity-cliff|Activity cliff]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/evaluation/ood-generalization|OOD generalization]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/reliability-diagram|Reliability diagram]]
- [[concepts/evaluation/uncertainty-estimation|Uncertainty estimation]]
- [[concepts/evaluation/robustness|Robustness]]
- [[concepts/evaluation/error-analysis|Error analysis]]
- [[concepts/evaluation/interpretability|Interpretability]]

## Research Links

- [[research/structure-based-ai/index|Structure-based AI]]
- [[research/protein-modeling/index|Protein modeling]]
- [[ai/generative-models|Generative models]]
