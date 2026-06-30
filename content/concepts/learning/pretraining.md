---
title: Pretraining
tags:
  - pretraining
  - representation-learning
  - machine-learning
---

# Pretraining

Pretraining learns reusable parameters or representations before adapting a model to a target task. It is the base pattern behind large language models, vision encoders, protein language models, molecular encoders, and many multimodal systems.

The generic two-stage view is:

$$
\theta_{\mathrm{pre}}
=
\arg\min_\theta
\mathbb{E}_{x\sim p_{\mathrm{source}}}
\left[
\mathcal{L}_{\mathrm{pre}}(f_\theta(x))
\right]
$$

followed by adaptation:

$$
\theta_{\mathrm{task}}
=
\arg\min_\theta
\mathbb{E}_{(x,y)\sim p_{\mathrm{target}}}
\left[
\mathcal{L}_{\mathrm{task}}(f_\theta(x),y)
\right],
\qquad
\theta_0=\theta_{\mathrm{pre}}
$$

## Common Objectives

- Autoregressive next-token prediction.
- Masked modeling over text, sequence, image patches, molecules, or proteins.
- Contrastive alignment between views or modalities.
- Denoising, reconstruction, or representation prediction.
- Multitask supervised pretraining over a broad source dataset.

## Source and Target Contract

Pretraining is incomplete without the source distribution and the downstream target distribution:

$$
p_{\mathrm{source}}(x)
\quad\rightarrow\quad
p_{\mathrm{target}}(x,y)
$$

The transfer claim depends on the relationship between them.

| Contract Part | Question |
| --- | --- |
| source examples | what raw objects were used before adaptation? |
| source objective | what signal was optimized during pretraining? |
| source filtering | what examples were removed, deduplicated, or upweighted? |
| target examples | what downstream task and split define success? |
| overlap rule | how are near-duplicates, homologs, scaffolds, templates, or documents removed? |
| adaptation rule | frozen probe, full fine-tune, adapter, LoRA, or retrieval use? |

If source and target overlap, the claim is no longer clean transfer. It may still be useful, but the evidence should be described as system performance rather than out-of-distribution generalization.

## Objective Families

| Family | Training Signal | Typical Evaluation |
| --- | --- | --- |
| autoregressive | predict next token or state | generation, perplexity, downstream adaptation |
| masked modeling | predict hidden token, patch, residue, atom, or region | linear probe, fine-tune, retrieval |
| contrastive | identify positive view among candidates | retrieval, neighborhood quality, transfer |
| joint embedding | predict target representation | representation evaluation and downstream tasks |
| denoising | recover clean object from corruption | generation, reconstruction, representation transfer |
| supervised multitask | many labeled source tasks | target transfer and robustness |

The objective can be useful even when it is not the final task. Record the evidence that connects pretraining loss to downstream behavior.

## Pretraining Contamination

For public notes, state the contamination boundary without listing private datasets or paths:

$$
\mathcal{D}_{\mathrm{pre}}
\cap
\mathcal{D}_{\mathrm{test}}
=
\varnothing
$$

In practice, exact set disjointness is often too weak. The relevant exclusion unit may be:

| Domain | Exclusion Unit |
| --- | --- |
| text or code | document, repository, benchmark item, paraphrase cluster |
| molecule | standardized molecule, scaffold, assay source |
| protein | sequence identity cluster, family, structure/template source |
| structure | PDB entry, chain, complex, template neighborhood |
| agent trace | task instance, tool state, benchmark item |

## Why It Matters

- Reduces labeled data requirements for downstream tasks.
- Makes architecture comparisons depend heavily on source data and objective.
- Can transfer useful structure, but can also transfer bias or shortcuts.
- In scientific AI, pretraining data boundaries can dominate claimed generalization.

## Checks

- What source distribution was used?
- What target signal did pretraining optimize?
- Does the pretraining objective match downstream evaluation?
- Could downstream test examples or close homologs/duplicates appear in pretraining?
- Is adaptation done by [[concepts/learning/linear-probing|probing]], [[concepts/learning/fine-tuning-protocol|full fine-tuning]], or parameter-efficient fine-tuning?
- Is the comparison controlled for source data scale, training tokens/examples, architecture, and adaptation budget?
- Is the claim representation quality, sample efficiency, robustness, or final system performance?

## Related

- [[concepts/learning/self-supervised-learning|Self-supervised learning]]
- [[concepts/learning/masked-modeling|Masked modeling]]
- [[concepts/learning/contrastive-learning|Contrastive learning]]
- [[concepts/learning/transfer-learning|Transfer learning]]
- [[concepts/learning/fine-tuning|Fine-tuning]]
- [[concepts/learning/representation-evaluation|Representation evaluation]]
- [[concepts/learning/linear-probing|Linear probing]]
- [[concepts/learning/fine-tuning-protocol|Fine-tuning protocol]]
- [[concepts/data/data-distribution|Data distribution]]
- [[concepts/evaluation/test-set-contamination|Test set contamination]]
- [[concepts/data/data-lineage|Data lineage]]
