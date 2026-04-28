# Experiment Protocol for EAI-CO

## Objective

Evaluate whether exploratory AI generation with multi-objective optimization improves social media advertising creatives compared with template-based, single-shot, open-source-only, and prompt-engineered baselines, using a publication-oriented design that combines full-benchmark automatic evaluation with stratified-subset human evaluation.

## Research Questions

1. Does EAI-CO improve automatic creative quality metrics compared with baselines?
2. Does EAI-CO improve human-rated attractiveness, clarity, audience relevance, and click intention on a stratified evaluation subset?
3. Which EAI-CO components contribute most to the final performance?
4. How strongly do automatic rewards correlate with human ratings?

## Study Design Rationale

This protocol follows a common SCI-style evaluation pattern for generative systems:

- **full-set automatic evaluation** for coverage, reproducibility, and method comparison;
- **subset human evaluation** for perceptual validity and practical relevance;
- **ablation studies** for mechanism-level evidence;
- **cost reporting** for realism and deployment relevance.

This design is appropriate because running large-scale human evaluation over every generated advertisement is expensive and rarely necessary when the paper's core claim concerns relative method quality rather than live platform CTR.

## Dataset Construction

1. Select 100 products from a public product or e-commerce image dataset.
2. Normalize metadata into:
   - product title;
   - category;
   - selling points;
   - original image path;
   - constraints;
   - preferred tone.
3. Pair each product with four audience segments:
   - students;
   - young professionals;
   - family users;
   - price-sensitive consumers.
4. Create 400 campaign briefs.

## Generation Conditions

For each product-audience task, generate one final advertisement from each condition:

- `B0_Template`
- `B1_SingleShot_API`
- `B2_OpenSource_Only`
- `B3_PromptEngineered_AI`
- `Ours_EAI_CO`

For EAI-CO, generate 3 rounds with 6 candidates per round and select the highest-reward final candidate.

## Innovation Claims to Support Experimentally

The experiment must support these non-trivial contributions:

1. **Exploratory generation instead of single-shot generation**: show that candidate search and refinement outperform one-pass prompting.
2. **Multi-objective creative optimization**: show that combining relevance, clarity, aesthetics, audience fit, diversity, and engagement is stronger than optimizing any single dimension indirectly.
3. **Audience-aware optimization**: show that explicit audience modeling improves human-perceived relevance and click intention.
4. **Diversity-preserving refinement**: show that exploration quality degrades when diversity is removed.

## Controlled Variables

- Same product metadata across all methods.
- Same audience segment definitions across all methods.
- Same output schema for all methods: headline, body, CTA, visual prompt, generated image.
- Same evaluation metrics across all methods.
- Same random seed list for repeated generation where applicable.

## Automatic Evaluation

Compute these metrics for every candidate:

- relevance;
- clarity;
- aesthetic;
- audience fit;
- brand safety;
- factuality penalty;
- diversity;
- predicted engagement;
- total reward;
- generation time;
- number of model calls.

For the final paper, replace mock scoring with:

- CLIP or similar image-text model for relevance;
- aesthetic predictor for visual quality;
- LLM/VLM rubric scoring for clarity and audience fit;
- moderation and factual consistency checks for safety;
- embedding distance for diversity;
- lightweight predictor trained from human preference labels or public engagement proxy data.

Report automatic results over the **entire benchmark**. This is the primary quantitative comparison layer for the paper.

## Human Evaluation

Human evaluation is conducted on a **stratified subset**, not on the full benchmark.

Recommended subset:

- 10 to 30 products depending on available labor;
- all 4 audience segments;
- 5 primary methods only in the first-stage human study;
- optional second-stage human study for selected ablations if reviewer feedback requires it.

Default paper configuration:

- 30 products;
- 4 audience segments;
- 5 methods;
- total 600 ad samples.

Participants:

- target approximately 20 to 60 participants;
- each ad receives at least 3 independent ratings for pilot publication support;
- prefer 5 or more independent ratings per ad for the final submission version;
- participants should be adults and consent to take part.

Rating dimensions on a 7-point scale:

1. attractiveness;
2. information clarity;
3. visual quality;
4. audience relevance;
5. click or purchase intention.

Pairwise preference:

- randomly pair EAI-CO outputs with baseline outputs from the same product-audience task;
- ask which ad the participant would be more likely to click;
- include an optional "no clear preference" choice only if required by the ethics review.

Human evaluation should be **blind to method identity** and, when possible, randomized in presentation order.

## Statistical Analysis

1. Inspect normality of metric distributions.
2. Use Friedman test for non-parametric repeated comparisons.
3. Use repeated-measures ANOVA if assumptions are met.
4. Use Wilcoxon signed-rank tests for pairwise method comparisons.
5. Apply Holm-Bonferroni correction.
6. Report effect size:
   - Cliff's delta for non-parametric pairwise comparisons;
   - partial eta squared for ANOVA.
7. Compute Spearman correlation between automatic reward and human click intention.

For submission-quality reporting, also include:

8. 95% confidence intervals for major primary metrics;
9. inter-rater reliability where feasible, such as Krippendorff's alpha or ICC;
10. a statement of which comparisons are confirmatory versus exploratory.

## Acceptance Criteria

EAI-CO is considered effective if:

- it significantly outperforms at least three baselines on total automatic reward;
- it significantly improves human-rated click intention over `B1_SingleShot_API` and `B3_PromptEngineered_AI`;
- ablation results show performance drops when iterative optimization or audience modeling is removed;
- safety and factuality penalties do not increase compared with baselines.

## Minimum Publishable Evidence

If compute and annotation resources are limited, the study is still academically defensible when it includes:

- full-benchmark automatic evaluation across all primary methods and ablations;
- human evaluation on a stratified subset of at least 10 products across all audience groups;
- at least 3 ratings per sample;
- paired significance testing for click intention and audience relevance;
- a clear statement that live CTR is not measured.

This is stronger than using automatic metrics alone and is consistent with common practice in generative-media evaluation.

## Reporting Rules

- Do not claim real CTR improvement without live platform data.
- Use "predicted engagement" and "human-rated click intention".
- Report model names, versions, access dates, seeds, prompts, and evaluation rubrics.
- Include failure cases and discuss model bias, hallucination, and brand-safety risks.
- Distinguish clearly among `automatic evaluation`, `proxy evaluation`, and `human evaluation`.
- If human evaluation covers a subset only, say so explicitly in the abstract, methods, and limitations.
