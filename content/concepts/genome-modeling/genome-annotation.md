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

## Annotation Sources

- Curated databases.
- Public reference annotations.
- Experimental assays.
- Computational predictions.
- Rule-based derived labels.

## Checks

- Which annotation source and version are used?
- Is the label measured, curated, or predicted by another model?
- Are annotation versions stable across train and test?
- Does metadata leak the target label?
- Are ambiguous or conflicting annotations handled explicitly?

## Related

- [[concepts/data/annotation-labeling|Annotation and labeling]]
- [[concepts/data/metadata-provenance|Metadata and provenance]]
- [[concepts/genome-modeling/genomic-region|Genomic region]]
- [[entities/genome|Genome]]
- [[concepts/evaluation/leakage|Leakage]]
