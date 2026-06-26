---
title: Split Unit
tags:
  - data
  - evaluation
  - leakage
---

# Split Unit

A split unit is the entity that must stay on one side of a train, validation, or test split. It is often broader than one example.

If $g(i)$ maps example $i$ to a split unit, then a leakage-safe split requires:

$$
g(i) = g(j)
\Rightarrow
\operatorname{split}(i)
=
\operatorname{split}(j)
$$

Examples sharing the same split unit should not be separated across train and test.

## Examples

| Task Type | Candidate Split Unit | Claim It Tests |
| --- | --- | --- |
| Molecule property | standardized molecule, scaffold, molecular cluster | generalization beyond near-duplicate chemistry |
| Molecule-target activity | scaffold, target, assay, source, target family | new chemistry, new target, or new assay context |
| Protein sequence | sequence identity cluster, protein family, domain family | generalization beyond close homologs |
| Structure-based modeling | protein family, ligand scaffold, complex pair, pocket family | new proteins, new ligands, or new binding contexts |
| Docking and pose ranking | receptor, ligand scaffold, complex, generated pose group | pose quality on unseen complexes rather than duplicate decoys |
| Genome sequence | locus, chromosome, region cluster, species, time/source batch | new genomic regions rather than overlapping windows |
| Document task | source document, author, collection, time period | new sources rather than memorized passages |
| User-event task | user, session, device, time block | new users or periods rather than event duplicates |

## Checks

- What entity defines the generalization claim?
- Is the split unit broader than the row identifier?
- Are near-duplicates grouped before splitting?
- Are preprocessing, scaling, and threshold choices fit only on train?
- Does the paper or benchmark report the split unit explicitly?
- Does validation use the same unit type as the final test claim?
- Are multiple unit types needed, such as scaffold and protein family?

## Multi-Axis Splits

Some datasets need more than one grouping axis. For protein-ligand modeling, a leakage-safe split may require grouping both sides:

$$
g(x_i)
=
(g_{\mathrm{protein}}(x_i), g_{\mathrm{ligand}}(x_i))
$$

The split rule should state whether it holds out new proteins, new ligands, new pairs, new assays, or a combination.

## Generalization Claims

A split unit is not only a data-engineering choice. It names the claim that the result can support.

| Split Rule | Supported Claim | Unsupported Claim |
| --- | --- | --- |
| random row split | interpolation over similar observed rows | novel scaffold, novel family, or deployment shift |
| scaffold split | new molecular series | new protein target unless targets are also held out |
| protein-family split | new protein families | new chemistry unless ligand similarity is controlled |
| time split | future-like deployment from past data | target or scaffold novelty unless grouped |
| source split | robustness across datasets or assays | broad biological generalization without label harmonization |
| paired holdout | new entity pair combinations | new entities unless each side is also held out |

When a paper reports a headline metric, read it as:

$$
\text{score} \mid \text{split unit}, \text{example unit}, \text{label source}
$$

The score is weaker or stronger depending on what the split unit excludes.

## Split Before Preprocessing

Grouping should happen before any operation that can collapse or leak information across train and test:

- Deduplicate and standardize identifiers according to the public contract.
- Build grouping keys such as scaffold, sequence cluster, target family, assay source, or locus.
- Assign groups to train, validation, and test.
- Fit scalers, vocabularies, thresholds, feature selectors, or imputers on training data only.
- Generate or cache representations with split-aware provenance.

## Related

- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/data/dataset-construction-checklist|Dataset construction checklist]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/leakage|Leakage]]
- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
