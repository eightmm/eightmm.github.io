---
title: Reward Modeling
tags:
  - reward-modeling
  - preference-optimization
  - evaluation
---

# Reward Modeling

Reward modeling learns a scoring function that approximates human, expert, user, or verifier preferences. It is often used when the true objective is difficult to write as a direct loss.

For pairwise preference data $(x,y_w,y_l)$, a common model is:

$$
P(y_w \succ y_l \mid x)
=
\sigma(r_\theta(x,y_w)-r_\theta(x,y_l))
$$

The reward model is trained with:

$$
\mathcal{L}_{\mathrm{RM}}
=
-\log \sigma(r_\theta(x,y_w)-r_\theta(x,y_l))
$$

where $y_w$ is preferred over $y_l$.

## Key Ideas

- A reward model turns qualitative judgment into an optimization target.
- The model can rank generated answers, agent trajectories, protein designs, molecular candidates, or tool-use outcomes if the comparison protocol is well defined.
- Reward models are proxies, so they can be exploited by policies trained too aggressively against them.
- Reward modeling should be paired with held-out human review, task metrics, or external verification.

## Practical Checks

- What exactly is being compared: final answer, full trajectory, code diff, molecule, pose, or experimental plan?
- Are preference labels consistent across annotators and task types?
- Does the reward model generalize outside the data distribution it was trained on?
- Are high-reward outputs also high quality under external evaluation?

## Related

- [[concepts/learning/preference-optimization|Preference optimization]]
- [[concepts/learning/reinforcement-learning|Reinforcement learning]]
- [[concepts/learning/policy-gradient|Policy gradient]]
- [[concepts/evaluation/metric|Metric]]
- [[agents/agent-evaluation|Agent evaluation]]
