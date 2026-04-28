# No-Human-Evaluation SCI Protocol for EAI-CO

## Objective

Evaluate whether exploratory AI generation with multi-objective optimization improves social media advertising creatives without relying on human raters, using a stronger automatic-and-proxy evaluation design suitable for SCI submission when human-subject collection is infeasible.

## Core Design

The study uses four evidence layers:

1. **full-benchmark automatic evaluation**
2. **component-wise ablation analysis**
3. **checklist-based LLM proxy judging**
4. **pairwise LLM judging with order-bias control**

## Research Questions

1. Does EAI-CO outperform strong baselines on full-benchmark automatic metrics?
2. Do proxy judges prefer EAI-CO to baselines under both absolute and pairwise evaluation?
3. Are the gains robust to prompt variation and pairwise order reversal?
4. Which internal components contribute most to the gains?

## Why This Is Still Publishable

When human evaluation is not available, the study remains academically defensible if it:

- separates `automatic evaluation` from `proxy evaluation`;
- avoids claiming real human preference;
- uses multiple proxy validation layers instead of a single opaque score;
- reports robustness, consistency, and limitations clearly.

## Dataset and Conditions

- 100 products
- 4 audience segments
- 400 product-audience tasks
- 5 primary methods
- 4 ablations

## Automatic Evaluation

Run the complete benchmark and report:

- relevance
- clarity
- aesthetic
- audience fit
- diversity
- brand safety
- factuality penalty
- predicted engagement
- total reward
- model calls
- latency

## Proxy Evaluation Layer 1: Checklist Judge

For each final ad, ask a judge model to score:

- headline clarity
- selling-point coverage
- audience alignment
- CTA strength
- brand safety
- visual-text consistency
- novelty
- overall score

Each ad is judged twice:

- default prompt
- strict prompt

Report:

- mean overall score
- mean checklist sum
- prompt-agreement rate
- correlation with automatic reward

## Proxy Evaluation Layer 2: Pairwise Judge

Compare `Ours_EAI_CO` against each baseline on the same task.

Judge twice:

- forward order: Ours vs Baseline
- reverse order: Baseline vs Ours

Report:

- Ours win rate
- tie rate
- order consistency
- confidence

This directly addresses position bias concerns.

## Robustness Requirements

To make the no-human design stronger:

1. use at least two prompt variants for checklist judging;
2. reverse pairwise order for every comparison;
3. report confidence intervals where possible;
4. include audience-wise breakdown;
5. include failure cases and judge disagreements.

## Claims Boundary

Allowed:

- predicted engagement improved
- proxy judge preference improved
- audience-conditioned creative quality improved

Not allowed:

- human preference improved
- real CTR improved
- user click behavior improved in deployment

## Minimum Submission-Grade Evidence

- full benchmark automatic results
- full ablation results
- checklist judge summary
- pairwise judge summary
- robustness and consistency analysis
- explicit limitations section about missing human evaluation
