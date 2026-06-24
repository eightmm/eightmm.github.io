---
title: OOD Generalization
tags:
  - evaluation
  - methodology
  - generalization
---

# OOD Generalization

Out-of-distribution (OOD) generalization measures how a model performs on data that differs systematically from training — new chemotypes, protein families, time periods, or domains. In-distribution accuracy routinely overstates real-world performance because deployment is rarely IID.

## Practical Checks

- Define the shift explicitly: structural, temporal, source, or covariate.
- Use grouped/structured splits (scaffold, family, time) to simulate the shift.
- Report the IID-to-OOD gap, not just one number.
- Check calibration under shift — confidence degrades before accuracy does.
- State the applicability domain: where the model is and is not expected to hold.

## Related

- [[concepts/evaluation/scaffold-split|Scaffold split]]
- [[concepts/evaluation/protein-family-split|Protein family split]]
- [[concepts/evaluation/calibration|Calibration]]
- [[concepts/evaluation/index|Evaluation]]
