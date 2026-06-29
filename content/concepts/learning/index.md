---
title: Learning Methods
tags:
  - learning
  - machine-learning
---

# Learning Methods

Learning method note는 특정 architecture와 독립적인 training objective와 representation-learning strategy를 설명합니다.

대부분의 learning method는 어떤 target signal $t(x)$를 만들고 어떤 loss를 최적화하는지로 설명할 수 있습니다.

$$
\min_\theta
\mathbb{E}_{x\sim p_{\mathrm{data}}}
\left[\mathcal{L}(f_\theta(x), t(x))\right]
$$

For supervised learning, $t(x)$ is a human or assay label. For self-supervised learning, $t(x)$ is derived from the input itself.

## Route Map

| Question | Start | Evidence Boundary |
| --- | --- | --- |
| what kind of objective is being optimized? | [Objective taxonomy](/concepts/learning/objective-taxonomy) | objective family, target construction, metric alignment |
| is the target externally labeled? | [Supervised learning](/concepts/learning/supervised-learning), [Semi-supervised learning](/concepts/learning/semi-supervised-learning) | label semantics, noise, split unit |
| is the target derived from the input? | [Self-supervised learning](/concepts/learning/self-supervised-learning) | augmentation, masking, collapse, transfer metric |
| is the model reused on a downstream task? | [Pretraining](/concepts/learning/pretraining), [Transfer learning](/concepts/learning/transfer-learning), [Fine-tuning](/concepts/learning/fine-tuning) | downstream protocol and data split |
| is representation quality the claim? | [Linear probing](/concepts/learning/linear-probing), [Representation evaluation](/concepts/learning/representation-evaluation) | frozen encoder, probe capacity, task diversity |
| is behavior learned from actions or preferences? | [Reinforcement learning](/concepts/learning/reinforcement-learning), [Preference optimization](/concepts/learning/preference-optimization) | reward source, off-policy data, evaluation loop |
| is the data distribution changing? | [Domain adaptation](/concepts/learning/domain-adaptation), [Active learning](/concepts/learning/active-learning) | source/target split and sampling policy |

## Learning Contract

Learning method note should state the training signal before discussing architecture.

$$
\mathcal{J}(\theta)
=
\mathbb{E}_{u\sim q}
\left[
\mathcal{L}_\theta(u)
\right]
$$

where $u$ is the training unit sampled by the method. Depending on the method, $u$ may be a labeled pair, masked view, augmented pair, noisy sample, preference pair, or trajectory.

| Contract part | Ask | Route |
| --- | --- | --- |
| training unit | what object is sampled by the objective? | [Example unit](/concepts/data/example-unit), [Task specification](/concepts/tasks/task-specification) |
| target construction | where does the target or feedback come from? | [Objective taxonomy](/concepts/learning/objective-taxonomy) |
| loss | what exact quantity is optimized? | [Loss function](/concepts/machine-learning/loss-function) |
| sampling policy | how are masks, views, negatives, prompts, or trajectories sampled? | [Augmentation policy](/concepts/learning/augmentation-policy), [Sampling strategy](/concepts/data/sampling-strategy) |
| model selection | which validation signal chooses checkpoint or threshold? | [Evaluation protocol](/concepts/evaluation/evaluation-protocol) |
| downstream evidence | what metric justifies the learning claim? | [Representation evaluation](/concepts/learning/representation-evaluation), [Metric selection](/concepts/evaluation/metric-selection) |

## Signal Map

| Signal source | Learning family | Main risk |
| --- | --- | --- |
| measured labels | supervised or semi-supervised learning | label semantics, noise, censoring |
| hidden input parts | masked modeling | target too local or leakage through context |
| augmented views | contrastive, JEPA, invariance learning | augmentation changes semantic identity |
| corrupted/noisy samples | denoising, diffusion, score, flow objectives | corruption process mismatched to downstream task |
| teacher outputs | distillation | teacher bias and hidden data access |
| demonstrations | imitation learning | behavior cloning distribution shift |
| preferences | reward modeling or preference optimization | preference proxy mismatch |
| environment reward | reinforcement learning | reward hacking and off-policy bias |
| newly queried labels | active learning | sampling policy changes data distribution |

## Task vs Objective vs Evidence

These should be recorded separately.

| Axis | Question | Example |
| --- | --- | --- |
| task | what behavior is needed? | rank ligands, classify images, predict structure |
| objective | what training signal is optimized? | cross-entropy, InfoNCE, masked likelihood, reward |
| architecture | what computation carries the signal? | Transformer, GNN, CNN, SSM |
| data | what examples and feedback are available? | labels, unlabeled corpus, demonstrations, preferences |
| evidence | how is success measured? | AUROC, retrieval recall, RMSD, task success |

An SSL pretraining loss can improve a downstream classifier, but the pretraining objective is not the downstream task. A preference objective can improve chat behavior, but it is not the same as agent task completion.

## Evaluation Boundary

Learning-method claims often fail because the evaluation budget changes with the method.

| Claim | Must Hold Fixed |
| --- | --- |
| better pretraining | downstream split, probe budget, fine-tuning budget, data access |
| better representation | frozen encoder, evaluator capacity, task suite, retrieval corpus |
| better fine-tuning | pretrained checkpoint, target data, search budget, early stopping |
| better domain adaptation | source/target split, label availability, target-domain leakage |
| better preference learning | preference data source, reward model, evaluation prompt/task set |
| better RL method | environment, reward, simulator version, rollout budget |

If the comparison changes architecture, data, objective, and evaluation at once, the note should say the claim is system-level rather than learning-method-specific.

## Supervision and Transfer

| Method | Use For |
| --- | --- |
| [Supervised learning](/concepts/learning/supervised-learning) | direct labels from humans, assays, annotations, or measurements |
| [Semi-supervised learning](/concepts/learning/semi-supervised-learning) | small labeled set plus larger unlabeled set |
| [Pretraining](/concepts/learning/pretraining) | learning reusable parameters before downstream adaptation |
| [Transfer learning](/concepts/learning/transfer-learning) | moving representations or weights to a target task |
| [Fine-tuning](/concepts/learning/fine-tuning), [Fine-tuning protocol](/concepts/learning/fine-tuning-protocol) | adapting pretrained models with explicit data and metric contracts |
| [Knowledge distillation](/concepts/learning/knowledge-distillation) | training a smaller or deployable model from teacher outputs |
| [Instruction tuning](/concepts/learning/instruction-tuning) | aligning language-model behavior to task instructions |
| [Curriculum learning](/concepts/learning/curriculum-learning) | changing sample difficulty over training |
| [Imitation learning](/concepts/learning/imitation-learning) | learning from demonstrated actions |

## Self-Supervised Learning

| Method | Core Target |
| --- | --- |
| [Self-supervised learning](/concepts/learning/self-supervised-learning) | target signal derived from the input |
| [Masked modeling](/concepts/learning/masked-modeling) | reconstruct hidden tokens, patches, residues, atoms, or regions |
| [Contrastive learning](/concepts/learning/contrastive-learning) | pull positive views together and push negatives apart |
| [JEPA](/concepts/learning/jepa) | predict latent representation of a target view |
| [Augmentation policy](/concepts/learning/augmentation-policy) | define invariances through view construction |
| [Representation collapse](/concepts/learning/representation-collapse) | failure mode where embeddings lose useful variation |
| [Representation evaluation](/concepts/learning/representation-evaluation) | measure transfer beyond the pretraining loss |

## Reinforcement and Preference Learning

| Method | Core Object |
| --- | --- |
| [Reinforcement learning](/concepts/learning/reinforcement-learning) | policy, state, action, reward, return |
| [Policy gradient](/concepts/learning/policy-gradient) | gradient estimator for expected return |
| [Reward modeling](/concepts/learning/reward-modeling) | learned proxy for human, assay, or environment preference |
| [Preference optimization](/concepts/learning/preference-optimization) | pairwise or listwise preference signal |

## Objective Lens

Learning method note는 training signal을 명시해야 합니다.

$$
\theta^\star
=
\arg\min_\theta
\mathbb{E}_{(x,t)\sim q}
\left[
\mathcal{L}_\theta(x,t)
\right]
$$

where $q$ defines how examples and targets are sampled. For papers, this means the method is incomplete until the note states:

- example unit and split unit;
- target construction rule;
- loss and metric;
- augmentation, masking, negative sampling, reward, or preference source;
- evaluation task used to justify representation or behavior claims.

## Domain Boundaries

| Domain | Extra check |
| --- | --- |
| molecules | scaffold leakage, chemical state, negative construction, activity cliffs |
| proteins | homolog split, family split, MSA/template leakage, residue indexing |
| structure models | coordinate frame, template use, pocket definition, geometric validity |
| LLMs | pretraining/test contamination, instruction data, preference source, tool access |
| agents | trace success, tool side effects, recovery, completion audit |

Move object-specific details to [[molecular-modeling/index|Computational Biology]] when the issue is protein, molecule, ligand, pocket, assay, or structure identity rather than the learning signal itself.

## Related

- [[concepts/tasks/index|Tasks]]
- [[concepts/learning/objective-taxonomy|Objective taxonomy]]
- [[concepts/architectures/index|Architectures]]
- [[concepts/evaluation/index|Evaluation]]
- [[agents/index|Agents]]
- [[molecular-modeling/protein-modeling|Protein modeling]]
- [[molecular-modeling/structure-based/index|Structure-based modeling]]
