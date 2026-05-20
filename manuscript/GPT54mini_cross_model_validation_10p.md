# GPT-5.4-mini Cross-Model Validation (10 Products)

## Setup

- Commercial model: `gpt-5.4-mini-2026-03-17`
- Products: `10`
- Audience segments: `4`
- Tasks: `40`
- Methods: `5` primary methods only
- Output directory: `outputs/real_generation_gpt54mini_10p`

## Main Result

The cross-model validation on `gpt-5.4-mini-2026-03-17` preserved the same ranking pattern observed in the local `Qwen2.5-7B-Instruct` main experiment:

- `Ours_EAI_CO = 0.9957`
- `B2_OpenSource_Only = 0.9751`
- `B1_SingleShot_API = 0.9736`
- `B3_PromptEngineered_AI = 0.9729`
- `B0_Template = 0.8396`

`Ours_EAI_CO` outperformed all baselines, and the gains were statistically significant:

- vs `B0_Template`: `+0.1561`, `p = 1.54e-08`
- vs `B1_SingleShot_API`: `+0.0221`, `p = 1.41e-03`
- vs `B2_OpenSource_Only`: `+0.0206`, `p = 6.42e-04`
- vs `B3_PromptEngineered_AI`: `+0.0228`, `p = 1.01e-03`

## Audience-wise Consistency

`Ours_EAI_CO` ranked first in all four audience groups:

- `family_users`: `0.9950`
- `price_sensitive_consumers`: `0.9938`
- `students`: `1.0000`
- `young_professionals`: `0.9939`

This supports the claim that the proposed optimization framework generalizes across persona-conditioned tasks even when the generative backbone is replaced with a commercial model.

## Interpretation

The commercial-model validation confirms the transferability of the proposed framework: the advantage of exploratory multi-objective optimization is not limited to the local open-source backbone.

One caveat should be stated explicitly in the manuscript: because `gpt-5.4-mini` is a stronger generator under the current evaluator, the reward values are compressed near the upper bound. Therefore, this experiment should be interpreted primarily as a ranking-preservation and transferability result rather than as the main source of effect-size separation.

## Reporting Recommendation

In the paper, this section should be presented as:

1. a fixed-subset supplementary experiment,
2. a cross-model validation study,
3. evidence that the proposed framework remains superior under a stronger commercial generator.
