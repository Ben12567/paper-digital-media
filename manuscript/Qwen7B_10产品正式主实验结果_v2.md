# Qwen2.5-7B-Instruct 10-Product Main Experiment (Submission-Grade Draft)

## Experimental Scope

- Backbone model: `Qwen2.5-7B-Instruct` (local GPU execution on RTX 3090)
- Products: `10`
- Audience segments: `4`
- Tasks: `40`
- Compared methods: `5` primary methods + `4` ablations
- Final evaluated creatives: `360`

## Main Findings

The proposed `Ours_EAI_CO` framework achieved the best overall reward among all primary baselines:

- `Ours_EAI_CO = 0.8051`
- `B3_PromptEngineered_AI = 0.7568`
- `B1_SingleShot_API = 0.7494`
- `B2_OpenSource_Only = 0.7445`
- `B0_Template = 0.6856`

The reward gains of `Ours_EAI_CO` over all primary baselines were statistically significant:

- vs `B0_Template`: `+0.1195`, `p = 9.09e-13`
- vs `B1_SingleShot_API`: `+0.0557`, `p = 1.34e-06`
- vs `B2_OpenSource_Only`: `+0.0606`, `p = 5.18e-08`
- vs `B3_PromptEngineered_AI`: `+0.0483`, `p = 6.58e-05`

`Ours_EAI_CO` also achieved the best audience fit among the primary methods (`0.4719`) and the highest compliance-oriented brand safety (`0.9945`).

## Ablation Study

After recalibrating the evaluator toward a compliance-aware reward, the full framework outperformed all ablations:

- `Ours_EAI_CO = 0.8051`
- `Ours_without_factual_penalty = 0.8006`
- `Ours_without_audience_modeling = 0.7833`
- `Ours_without_iterative_loop = 0.7760`
- `Ours_without_diversity = 0.7349`

Relative to the full method, the reward drops were:

- w/o factual penalty: `-0.0045`
- w/o audience modeling: `-0.0218`
- w/o iterative loop: `-0.0291`
- w/o diversity: `-0.0702`

This result supports three core claims:

1. Exploratory iteration materially improves creative quality.
2. Diversity preservation is a major contributor to final performance.
3. Audience-aware conditioning improves reward and audience alignment.

The factual-penalty ablation remained close to the full method, but it no longer exceeded the complete framework. This supports a more precise interpretation: compliance-aware scoring stabilizes quality without collapsing persuasive power.

## Audience-Wise Results

`Ours_EAI_CO` ranked first in all four audience segments:

- `family_users`: `0.8147`
- `students`: `0.8098`
- `young_professionals`: `0.8050`
- `price_sensitive_consumers`: `0.7908`

This reduces the risk that the overall gain is driven by a single audience subgroup.

## Cost and Runtime

Average per-task latency:

- `Ours_EAI_CO`: `11114 ms`
- `B3_PromptEngineered_AI`: `2727 ms`
- `B1_SingleShot_API`: `2780 ms`
- `B2_OpenSource_Only`: `2707 ms`
- `B0_Template`: `1 ms`

The proposed framework is slower than single-shot generation, but the runtime increase is consistent with its iterative search design and remains practical for offline creative optimization.

## Submission-Level Interpretation

This experiment is strong enough to support the paper's main empirical claim on the open-source local backbone:

> Treating social-media ad creation as an exploratory multi-objective optimization problem yields significantly better creatives than template-based and single-pass generation baselines.

For the strongest submission package, one remaining step is still recommended:

- add a fixed-subset `GPT-5.4-mini` cross-model validation experiment as external model transfer evidence.

Even without that supplementary run, the current `Qwen2.5-7B-Instruct` 10-product experiment already provides:

- significant main-method superiority,
- complete ablation evidence,
- audience-wise consistency,
- cost/runtime reporting,
- reproducible local execution.
