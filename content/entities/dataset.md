---
title: Dataset
tags:
  - entities
  - dataset
---

# Dataset

A dataset is a curated collection of examples, labels, metadata, and splits used to train or evaluate models.

For chem-bio datasets, one row should usually preserve the [[entities/target-assay-label|Target-assay-label contract]] instead of collapsing the row into only `molecule -> label`.

A useful dataset note treats the dataset as an object with lineage:

$$
\mathcal{D}
=
(\mathcal{S}, \mathcal{U}, \mathcal{X}, \mathcal{Y}, \mathcal{M}, \Pi)
$$

where $\mathcal{S}$ is source records, $\mathcal{U}$ is the example-unit set, $\mathcal{X}$ is model input, $\mathcal{Y}$ is labels or targets, $\mathcal{M}$ is metadata, and $\Pi$ is the split policy.

## Dataset Object

| Part | Question |
| --- | --- |
| Source | Where did the raw records come from, and can the source be cited publicly? |
| Schema | What fields, identifiers, units, and relationships exist? |
| Example unit | What is exactly one example? |
| Label semantics | What does the target mean, and under what measurement process? |
| Metadata | What context is required to interpret each example? |
| Split policy | What grouping prevents leakage for the intended claim? |
| Preprocessing | What transformations produce model-ready inputs? |
| Artifact boundary | Which files, hashes, versions, or scripts define this dataset version? |

## Example Unit vs Split Unit

The row identifier is rarely enough. A dataset should state both:

$$
u_i = \text{example unit}, \qquad g(u_i)=\text{split unit}
$$

If two examples share the same split unit, they should stay on the same side of train, validation, and test for that claim.

| Task | Example Unit | Split Unit |
| --- | --- | --- |
| molecule property | standardized molecule or measurement row | scaffold, molecular cluster, source |
| target-conditioned activity | molecule-target-assay row | scaffold, protein family, assay/source |
| protein sequence task | sequence, domain, residue, or family record | sequence cluster or protein family |
| docking or pose task | receptor-ligand pair, pose, or complex | receptor, ligand scaffold, complex, pocket family |
| genome sequence task | window, region, variant, or locus | chromosome, locus, species, source batch |

The split unit names the generalization claim. A random row split and a scaffold split are different scientific statements even if the model and metric are identical.

## Label and Metadata Boundary

For supervised datasets, label meaning should not be inferred from a column name alone.

| Label Issue | Required Context |
| --- | --- |
| binary activity | endpoint, threshold, unit, source, missing-label policy |
| affinity or potency | unit, transformation, censoring, assay protocol |
| pose correctness | reference structure, tolerance, failed-pose handling |
| generated-object validity | validity rule, invalid denominator, filtering policy |
| sequence annotation | reference database, version, evidence level, region definition |

For chem-bio rows, the label often belongs to a tuple:

$$
y_i = h(e_i, c_i)
$$

where $e_i$ is the main entity and $c_i$ is context such as target, assay, pocket, source, time, construct, or measurement protocol.

## Dataset Version

A dataset version is not just a file name. It should include enough information to reconstruct or audit the claim:

| Version Field | Example |
| --- | --- |
| source snapshot | public dataset release, paper supplement, database date |
| filtering rule | allowed records, excluded records, failure handling |
| standardization | molecule normalization, sequence filtering, structure preparation |
| split rule | grouping key, validation policy, held-out claim |
| preprocessing artifacts | vocabulary, scaler, featurizer, conformer generator, graph builder |
| checksums or manifests | public-safe hashes or manifest shape |
| limitations | coverage gaps, bias, label noise, domain shift |

## Why It Matters

- Splits, duplicate handling, and label provenance shape what a benchmark measures.
- Molecular datasets often mix [[entities/molecule|molecules]], [[entities/protein|proteins]], assays, and structures.
- Clear dataset notes make leakage risks and task definitions easier to audit.
- Dataset lineage explains how the dataset version was produced from source records.
- Dataset cards keep model claims from drifting beyond the data's supported scope.

## Public Dataset Note Shape

Use this shape when writing about a public dataset:

| Section | Include |
| --- | --- |
| Purpose | task family and intended use |
| Public source | citation, license, release, and `to verify` for missing metadata |
| Example unit | row/entity/pair/pose/region definition |
| Schema | identifiers, fields, units, optional fields |
| Labels | meaning, unit, transform, censoring, missingness, weak labels |
| Splits | split unit, validation rule, leakage checks |
| Preprocessing | transformations fit on train only, versioned featurization |
| Evaluation | supported metrics and unsupported claims |
| Limitations | sampling bias, label noise, assay/source drift, applicability domain |

Do not publish private raw paths, collaborator-only datasets, internal labels, unpublished metrics, or unreleased split files. If a dataset is private, write the general dataset pattern rather than the private artifact.

## Checks

- What is one training example, and what is the target label?
- Is the example unit different from the row identifier?
- What split unit matches the intended generalization claim?
- Are duplicate molecules, protein families, or near-identical structures separated?
- Does the split match the intended generalization setting?
- Are missing labels, censored measurements, and assay metadata handled explicitly?
- Is the split key the same entity that defines the intended generalization claim?
- Is label provenance kept with each row?
- Does each row keep target, assay, unit, threshold, censoring, and source context when relevant?
- Are preprocessing artifacts fit only on training data?
- Can the public note support the claim without exposing private sources or unpublished results?

## Related

- [[entities/entity-relation-map|Entity relation map]]
- [[entities/target-assay-label|Target-assay-label contract]]
- [[concepts/data/index|Data]]
- [[concepts/data/dataset-card|Dataset card]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[concepts/data/dataset-split-contract|Dataset split contract]]
- [[concepts/data/dataset-construction-checklist|Dataset construction checklist]]
- [[concepts/data/preprocessing-contract|Preprocessing contract]]
- [[concepts/data/label-semantics|Label semantics]]
- [[concepts/data/data-curation|Data curation]]
- [[concepts/data/data-lineage|Data lineage]]
- [[concepts/data/annotation-labeling|Annotation and labeling]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[entities/bioactivity-label|Bioactivity label]]
- [[concepts/data/benchmark|Benchmark]]
- [[concepts/machine-learning/data-preprocessing|Data preprocessing]]
- [[concepts/evaluation/train-validation-test-split|Train/validation/test split]]
- [[concepts/evaluation/assay-harmonization|Assay harmonization]]
- [[concepts/evaluation/activity-cliff|Activity cliff]]
- [[concepts/evaluation/applicability-domain|Applicability domain]]
- [[entities/assay|Assay]]
- [[entities/molecule|Molecule]]
- [[entities/sequence|Sequence]]
- [[entities/structure|Structure]]
