---
title: Computational Biology Paper Claim Patterns
tags:
  - molecular-modeling
  - computational-biology
  - papers
  - evaluation
---

# Computational Biology Paper Claim Patterns

Computational biology papers often look different on the surface but reuse a small set of claim patterns. Use this page before writing a paper note when the abstract mentions molecules, proteins, docking, conformers, complexes, or genome sequence modeling.

The generic pattern is:

$$
\text{claim}
=
(\text{object},\ \text{task},\ \text{representation},\ \text{objective},\ \text{evidence})
$$

Do not route a paper only by model name. A Transformer for protein sequences, a GNN for molecular graphs, and a diffusion model for ligand poses need different object and evaluation contracts.

## Pattern Map

| Pattern | Object | Typical Claim | Primary Route |
| --- | --- | --- | --- |
| Molecular property prediction | molecule or ligand | better scalar/class prediction | [Molecules](/molecular-modeling/molecules), [Property prediction](/concepts/tasks/property-prediction) |
| Target-conditioned activity | molecule, target, assay | better bioactivity or affinity prediction | [Data and evaluation](/molecular-modeling/data-evaluation), [Target-assay-label contract](/entities/target-assay-label) |
| Protein representation learning | protein sequence or structure | better transfer to downstream protein tasks | [Proteins](/molecular-modeling/proteins), [Self-supervised learning](/concepts/learning/self-supervised-learning) |
| Protein structure prediction | protein sequence, MSA, template, structure | better coordinates or contact geometry | [Protein modeling concepts](/concepts/protein-modeling), [Coordinate modeling contract](/concepts/geometric-deep-learning/coordinate-modeling-contract) |
| Docking or pose generation | protein-ligand complex | better pose, ranking, or enrichment | [Docking](/molecular-modeling/docking), [SBDD concepts](/concepts/sbdd) |
| Structure-conditioned generation | pocket, ligand, complex | valid and useful molecules under context | [Generative models](/ai/generative-models), [Structure-based modeling](/molecular-modeling/structure-based) |
| Conformer generation | molecule, conformer ensemble | plausible low-energy 3D structures | [Conformer](/concepts/molecular-modeling/conformer), [Geometry](/molecular-modeling/geometry) |
| Protein design | sequence, backbone, fold, binder, function | generated designs satisfy structural or functional constraints | [Protein design](/concepts/generative-models/protein-design), [Protein modeling](/molecular-modeling/protein-modeling) |
| Genome sequence modeling | sequence window, variant, region | better sequence or variant-level prediction | [Genome](/molecular-modeling/genome), [Genome modeling concepts](/concepts/genome-modeling) |

## Property Prediction

The basic supervised form is:

$$
\hat{y}
=
f_\theta(r(m)),
\qquad
\hat{\theta}
=
\arg\min_\theta
\frac{1}{n}\sum_{i=1}^{n}
\mathcal{L}
\left(
f_\theta(r(m_i)), y_i
\right)
$$

- $m_i$: molecule or ligand.
- $r(m_i)$: SMILES, graph, fingerprint, descriptor, conformer, or embedding.
- $y_i$: property label with unit, endpoint, and direction.
- $\mathcal{L}$: classification, regression, ranking, or censored-label objective.

Minimum evidence:

| Required | Why |
| --- | --- |
| Chemical state policy | salt, tautomer, protonation, stereo, and conformer choices can change identity |
| Split unit | random split may overstate scaffold or series generalization |
| Baseline | fingerprint/tree or simple descriptor baselines catch weak representation claims |
| Metric alignment | RMSE, AUROC, PR-AUC, enrichment, and calibration answer different questions |

## Target-Conditioned Activity

Bioactivity rows usually depend on target and assay context:

$$
\hat{y}
=
f_\theta(r(m), r(t), r(a))
$$

- $m$: molecule or ligand.
- $t$: target, protein, pocket, family, or construct.
- $a$: assay, endpoint, condition, source, threshold, or unit.
- $y$: measured activity, affinity, binary hit, ranking label, or censored value.

Do not collapse this into only `molecule -> label` unless the target and assay are truly fixed.

Minimum evidence:

| Required | Why |
| --- | --- |
| Target-assay-label contract | same molecule can have different labels under different targets or assays |
| Split on both axes | ligand-only or protein-only split may not test pair generalization |
| Assay harmonization | pooled assay values can mix incompatible measurement processes |
| Applicability domain | prediction reliability can fail on new chemotype, target family, or assay source |

## Protein Representation Learning

Many protein papers optimize a pretraining objective and evaluate transfer:

$$
z_i
=
g_\theta(s_i),
\qquad
\hat{y}_i
=
h_\phi(z_i)
$$

- $s_i$: protein sequence, MSA, residue graph, or structure-derived representation.
- $g_\theta$: pretrained encoder.
- $h_\phi$: probe, fine-tuned head, retrieval scorer, or task-specific model.
- $y_i$: downstream label such as family, function, stability, contact, or binding.

Minimum evidence:

| Required | Why |
| --- | --- |
| Representation unit | residue, sequence, domain, chain, complex, and pocket embeddings are not interchangeable |
| Adaptation budget | linear probe, kNN, frozen head, and full fine-tuning test different claims |
| Protein split | random sequence splits can leak homologous families |
| Downstream metric | representation quality is task- and evaluator-dependent |

## Docking, Pose, and Screening

Pose claims depend on coordinates and inference-time information:

$$
S_\theta(p, l, X)
\rightarrow
\text{rank or pose decision}
$$

- $p$: protein, receptor, or pocket context.
- $l$: ligand.
- $X$: generated or evaluated ligand coordinates in the pocket.
- $S_\theta$: score used for pose selection, affinity ranking, or virtual screening.

Minimum evidence:

| Required | Why |
| --- | --- |
| Pocket definition | ligand-defined pockets can leak evaluation information |
| Coordinate source | experimental, predicted, apo, holo, and template-derived structures support different claims |
| Pose metric | RMSD, clash, interaction recovery, and plausibility measure different failures |
| Ranking metric | pose success does not imply affinity ranking or enrichment |
| Template and analog leakage | close structures or analogs can make the test task easier |

## Generative Molecules and Complexes

Generative claims should separate model objective from utility evaluation:

$$
x
\sim
p_\theta(x \mid c),
\qquad
\text{utility}(x,c)
\neq
\log p_\theta(x \mid c)
$$

- $x$: molecule, conformer, pose, sequence, backbone, or complex.
- $c$: condition such as target, pocket, property, scaffold, fold, or function.
- $p_\theta$: autoregressive, diffusion, flow, VAE, GAN, or search distribution.

Minimum evidence:

| Required | Why |
| --- | --- |
| Validity rule | syntactic validity, chemical validity, structural plausibility, and biological utility differ |
| Novelty and diversity | high average score can hide near-duplicates or mode collapse |
| Conditional satisfaction | generated samples must satisfy the stated condition, not only look plausible |
| Filtering policy | post-generation filters can own the final performance claim |
| Utility metric | docking score, affinity proxy, property predictor, or experimental assay has a scope boundary |

## Protein Design

Protein design may generate sequence, structure, or both:

$$
s, X
\sim
p_\theta(s, X \mid c)
$$

- $s$: amino-acid sequence.
- $X$: backbone, side-chain, complex, or generated coordinate structure.
- $c$: fold, motif, target, binding site, function, or stability condition.

Minimum evidence:

| Required | Why |
| --- | --- |
| Design target | fold, binder, enzyme, scaffold, or sequence recovery are different claims |
| Structure validation | predicted foldability is not the same as experimental function |
| Novelty check | training-template or family similarity can inflate design success |
| Experimental boundary | in-silico scores should not be written as wet-lab validation |

## Genome Sequence Modeling

Genome notes in this site are limited to sequence, region, k-mer, and variant-level modeling:

$$
\hat{y}
=
f_\theta(r(g_{a:b}), c)
$$

- $g_{a:b}$: genomic sequence window or region.
- $r(\cdot)$: tokens, k-mers, embedding, annotation features, or variant context.
- $c$: organism, coordinate system, strand, annotation source, or task condition.
- $y$: sequence label, region label, or variant-effect target.

Minimum evidence:

| Required | Why |
| --- | --- |
| Coordinate system | region definitions depend on reference build and indexing convention |
| Split unit | random windows can leak nearby sequence or homologous regions |
| Annotation source | labels can reflect database and pipeline artifacts |
| Scope boundary | this site does not expand into broad omics or clinical interpretation by default |

## Final Routing Checklist

- Does the paper's strongest claim belong to object/domain, AI method, Math objective, benchmark, systems, or agent workflow?
- Is the biological object explicit before the model name?
- Is the raw object separated from model-ready representation?
- Is the loss written with sampled unit and distribution?
- Is the evaluation metric tied to split, baseline, uncertainty, and failure mode?
- Are generated or predicted structures checked under the correct coordinate and symmetry contract?
- Are public artifacts and missing metadata marked honestly?

## Related

- [[molecular-modeling/paper-intake|Computational Biology paper intake]]
- [[concepts/ai-computational-biology-math-contract|AI Computational Biology Math contract]]
- [[molecular-modeling/data-evaluation|Data and evaluation]]
- [[ai/paper-intake|AI paper intake]]
- [[math/formula-intake|Formula intake]]
- [[papers/workflows/claim-routing|Claim routing]]
- [[papers/workflows/ai-molecular-math-paper-template|AI Computational Biology Math paper template]]
- [[concepts/coverage-matrix|Coverage matrix]]
