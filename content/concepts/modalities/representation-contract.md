---
title: Representation Contract
tags:
  - modalities
  - representation
  - data
---

# Representation Contract

A representation contract records how a raw object becomes the model input used for training and evaluation. It is the bridge between entity, modality, architecture, loss, metric, and leakage checks.

The core map is:

$$
x_{\mathrm{raw}}
\xrightarrow{\phi}
r
\xrightarrow{f_\theta}
\hat{y}
$$

where $x_{\mathrm{raw}}$ is the raw object, $\phi$ is preprocessing or featurization, $r$ is the model-ready representation, and $\hat{y}$ is the model output.

## Contract Fields

| Field | Question |
| --- | --- |
| Raw object | What is the entity before preprocessing: text, image, molecule, protein, pocket, complex, graph, or sequence? |
| Representation | What does the model see: tokens, tensor, graph, fingerprint, conformer, coordinate set, embedding, or fused view? |
| Unit | What is one token, node, edge, residue, atom, frame, candidate, or example? |
| Shape and axes | What are the batch, token, node, edge, coordinate, channel, head, or candidate axes? |
| Featurizer | Is $\phi$ deterministic, learned, stochastic, cached, versioned, or data-dependent? |
| Invariance | Should the representation preserve order, permutation, translation, rotation, time, units, or metadata? |
| Missingness | What happens when fields, modalities, coordinates, labels, or alignments are missing? |
| Output | What output space does this representation support? |
| Loss and metric | Which loss trains the model, and which metric evaluates the output? |
| Split | Which unit must be held out to support the generalization claim? |
| Leakage | Does $\phi$ use labels, target poses, future information, test statistics, or deployment-unavailable context? |

## AI Use

For AI architecture notes, the representation contract explains why an architecture is appropriate.

| Representation | Likely Architecture Bias | Check |
| --- | --- | --- |
| Token sequence | Transformer, RNN, state-space model | tokenization, context length, positional encoding |
| Image patches or feature maps | CNN, ViT, U-Net | locality, resolution, augmentation policy |
| Graph | GNN, Graph Transformer | node/edge definition, graph construction leakage |
| Set | Deep Sets, Set Transformer, Perceiver | permutation behavior and pooling |
| Coordinates | geometric GNN, equivariant model, diffusion/flow | transformation group and coordinate-frame leakage |
| Embedding | MLP, retrieval model, linear probe | frozen vs trainable encoder and evaluator capacity |

## Molecular Modeling Use

| Object | Representation Questions |
| --- | --- |
| Molecule | SMILES, graph, fingerprint, descriptors, conformer, tautomer/protonation/stereo policy |
| Protein | sequence tokens, MSA, residue embedding, residue graph, coordinates, pocket representation |
| Ligand pose | conformer source, coordinate frame, atom mapping, symmetry handling |
| Protein-ligand complex | pocket definition, ligand state, interaction graph, distance features, pose availability |
| Assay record | target, assay, endpoint, unit, threshold, censoring, source |

## Formula View

If the representation is stochastic or learned, write:

$$
r \sim q_\psi(r \mid x_{\mathrm{raw}}, c)
$$

where $c$ is context such as pH, assay, target, crop, alignment, retrieval corpus, or conformer protocol. If $c$ is unavailable at inference, evaluation may be invalid.

## Checks

- Is the raw object separate from the model input?
- Are tensor shapes and axis meanings explicit?
- Is the representation identical across train, validation, test, and inference where it should be?
- Are featurizer versions, random seeds, caches, and failure policies recorded?
- Does the representation preserve the information required by the task?
- Does the representation include information unavailable at deployment time?
- Is the split defined before or after representation caching?
- Does the loss and metric match the output space implied by the representation?
- Does the architecture match the representation structure rather than only the paper's keyword?

## Related

- [[concepts/modalities/modality-representation|Modality representation]]
- [[concepts/modalities/modality-task-map|Modality-task map]]
- [[concepts/math/tensor-shape-notation|Tensor shape notation]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/molecular-modeling/molecular-featurization-contract|Molecular featurization contract]]
- [[concepts/protein-modeling/protein-representation|Protein representation]]
- [[concepts/architectures/architecture-selection|Architecture selection]]
- [[concepts/machine-learning/objective-metric-alignment|Objective-metric alignment]]
- [[concepts/evaluation/claim-evidence-boundary|Claim-evidence boundary]]
