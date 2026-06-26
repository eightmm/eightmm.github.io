---
title: Energy Minimization
tags:
  - molecular-modeling
  - optimization
  - geometry
---

# Energy Minimization

Energy minimization adjusts molecular coordinates to reduce an energy function, often from a [[concepts/molecular-modeling/force-field|force field]]. It is used to relax conformers, remove clashes, refine poses, or prepare structures before downstream modeling.

The basic objective is:

$$
X^\*
=
\arg\min_X E_{\pi}(X)
$$

where $X$ is the coordinate matrix and $E_{\pi}$ is the energy under protocol $\pi$.

## Role in Modeling

Energy minimization can be part of preprocessing, postprocessing, evaluation, or baseline construction. These are different claims:

| Use | Meaning | Risk |
| --- | --- | --- |
| Preprocessing | clean input geometry before a model sees it | hidden information can enter through the protocol |
| Postprocessing | relax model output | performance may come from minimization, not the model |
| Evaluation filter | remove invalid structures before scoring | invalid-rate may be hidden |
| Baseline | compare against a classical optimization method | initialization and constraints dominate |

## Optimization View

Gradient-based minimization follows:

$$
X_{t+1}
=
X_t
-
\eta_t \nabla_X E_{\pi}(X_t)
$$

Some workflows use constraints or restraints:

$$
\min_X
E_{\pi}(X)
+
\lambda R(X, X_0)
$$

where $R$ penalizes movement away from a reference geometry $X_0$.

## Checks

- Is minimization applied to inputs, outputs, or both?
- What atoms are free, fixed, or restrained?
- Is the receptor included or is the ligand minimized alone?
- Does minimization happen before or after evaluation metrics are computed?
- Are invalid pre-minimized outputs counted?
- Is the same minimization protocol used for all compared methods?

## Failure Modes

- A model appears better because outputs are heavily postprocessed.
- RMSD improves while protein-ligand contacts or chemistry become worse.
- Clash removal hides that the generative model produced impossible structures.
- Ligand-only minimization creates a conformer incompatible with the binding pocket.
- The energy function prefers a geometry that does not match the experimental or assay context.

## Related

- [[concepts/molecular-modeling/force-field|Force field]]
- [[concepts/molecular-modeling/conformer|Conformer]]
- [[concepts/sbdd/pose-generation|Pose generation]]
- [[concepts/sbdd/pose-quality|Pose quality]]
- [[concepts/machine-learning/optimization|Optimization]]
- [[math/calculus-gradients|Calculus and gradients]]
