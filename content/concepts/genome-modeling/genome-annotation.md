---
title: Genome Annotation
tags:
  - genome
  - annotation
  - data
---

# Genome Annotation

Genome annotation assigns labels or metadata to genomic regions. In machine learning, annotation can be a training label, auxiliary feature, evaluation target, or provenance field.

An annotated region can be represented as:

$$
(r_i, y_i, m_i)
$$

where $r_i$ is the region, $y_i$ is the annotation label, and $m_i$ records source metadata.

The annotation is meaningful only relative to a reference system:

$$
a_i =
(\text{assembly}, \text{coordinate}, \text{region}, \text{label}, \text{source}, \text{version})
$$

Changing the assembly, annotation source, or version can change the task even if the sequence window looks similar.

## Annotation Sources

| Source Type | Use | Risk |
| --- | --- | --- |
| curated database | stable labels and metadata | version drift and curation bias |
| public reference annotation | genes, transcripts, regions, functional labels | assembly and coordinate mismatch |
| experimental assay | measured signal for regions or variants | assay/source-specific labels |
| computational prediction | weak label or auxiliary feature | teacher-model leakage |
| rule-based derived label | deterministic preprocessing target | hidden rule encodes target information |

## Label Roles

The same annotation can play different roles in a model pipeline.

| Role | Example Question |
| --- | --- |
| target label | can the model predict the annotation from sequence? |
| input feature | can annotation context improve another task? |
| filter | which regions are included or excluded? |
| grouping key | which regions must stay together across splits? |
| provenance field | where did this label come from? |

Do not mix these roles silently. If an annotation is used for filtering or grouping, it may affect the evaluation claim even if it is not a model input.

## Coordinate Contract

Genome annotations need a coordinate contract:

$$
r = (\text{assembly}, \text{chromosome}, s, e, \text{strand})
$$

where $s$ and $e$ are start and end coordinates. A public note should state whether intervals are 0-based or 1-based when that matters.

| Field | Why |
| --- | --- |
| assembly | maps sequence to coordinates |
| chromosome or contig | defines the coordinate namespace |
| start/end convention | prevents off-by-one label shifts |
| strand | affects sequence extraction and reverse complement |
| annotation version | prevents label drift across releases |

## Split and Leakage

Annotation tasks can leak through overlapping windows, near-duplicate regions, shared evidence sources, and annotation-derived filters.

$$
\operatorname{overlap}(r_{\mathrm{train}}, r_{\mathrm{test}})=0
$$

is a minimum condition for region-level generalization, not a complete guarantee. For broader claims, group by region family, chromosome, source, species, or annotation evidence as needed.

| Leakage Path | Check |
| --- | --- |
| overlapping sequence windows | split regions before window extraction |
| same gene or locus across splits | group by locus or gene when relevant |
| labels derived from same source | separate source or version if claiming source generalization |
| annotation used as filter | record filtering before interpreting the metric |
| reverse-complement duplicates | group both orientations |

## Checks

- Which annotation source and version are used?
- Is the label measured, curated, or predicted by another model?
- Are annotation versions stable across train and test?
- Does metadata leak the target label?
- Are ambiguous or conflicting annotations handled explicitly?
- What genome assembly and coordinate convention are used?
- Is the annotation a target, feature, filter, grouping key, or provenance field?
- Are overlapping windows, adjacent regions, and reverse complements split safely?
- Is the public claim limited to sequence/region/variant modeling rather than broad omics?

## Related

- [[concepts/data/annotation-labeling|Annotation and labeling]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/genome-modeling/genomic-region|Genomic region]]
- [[concepts/genome-modeling/variant-effect-prediction|Variant effect prediction]]
- [[concepts/data/example-unit|Example unit]]
- [[concepts/data/split-unit|Split unit]]
- [[entities/dataset|Dataset]]
- [[entities/genome|Genome]]
- [[concepts/evaluation/leakage|Leakage]]
