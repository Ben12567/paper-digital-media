# Exploratory Artificial Intelligence for Automated Social Media Advertising Content Creation and Multi-objective Creative Optimization

## Abstract

Generative artificial intelligence has accelerated digital media content production, but most practical workflows still rely on single-shot generation, where candidate creatives are produced once and then manually selected. This limits personalization, systematic exploration, and measurable optimization. This paper proposes **EAI-CO**, an exploratory artificial intelligence framework for automated social media advertising content creation and multi-objective creative optimization. EAI-CO encodes campaign briefs, generates diverse text-image advertising candidates, evaluates them with automatic multi-objective metrics, and iteratively improves candidates through exploration and selection. The study focuses on social media image-text advertisements and evaluates EAI-CO against template-based, single-shot API-based, open-source-only, and prompt-engineered baselines. Evaluation combines automatic metrics, ablation studies, and human preference ratings. The expected contribution is a reproducible framework showing how exploratory generation and iterative optimization can improve predicted engagement, audience fit, information clarity, and human-rated click intention without claiming real platform CTR improvement.

**Keywords:** exploratory artificial intelligence; generative AI; social media advertising; automated content creation; creative optimization; multi-objective evaluation

## 1. Introduction

Digital media advertising requires large volumes of personalized content across products, audiences, and platforms. Social media campaigns often need multiple variants of headlines, captions, calls to action, and visual styles. Traditional manual creative production is costly and slow, while template-based automation struggles to produce diverse and audience-sensitive content.

Generative AI provides a new route for automated advertising content creation, but many workflows use it as a one-step generator. A single prompt produces a single or small set of outputs, and the burden of evaluation remains with human designers or marketers. This weakens repeatability and makes it difficult to link generation with measurable optimization. Recent studies on generative AI in marketing emphasize the need for new performance indicators, monitoring mechanisms, and human-centered evaluation when AI systems participate in content production.

This paper argues that AI-based digital media content creation should be treated as an exploratory optimization problem rather than a single-shot generation task. The proposed EAI-CO framework allows an AI system to generate diverse candidates, evaluate them using multiple objectives, preserve useful diversity, penalize unsafe or factually unsupported content, and refine candidates over multiple rounds.

The contributions are:

1. A five-module exploratory AI framework for social media advertising content creation.
2. A multi-objective evaluation design integrating relevance, clarity, aesthetics, audience fit, diversity, safety, and predicted engagement.
3. A reproducible experimental protocol combining full-benchmark automatic evaluation, mechanistic ablations, and stratified-subset human evaluation.
4. An empirical validation strategy that tests whether automatic optimization signals align with human-rated ad quality and click intention.

## 2. Related Work

### 2.1 Generative AI in Marketing and Advertising

Generative AI is reshaping marketing content production by enabling rapid creation of copy, images, campaign variants, and personalized communication. However, marketing applications require more than fluent generation. They require brand consistency, factual reliability, audience relevance, and measurable effectiveness. Prior work highlights that marketers need new KPIs and governance mechanisms when automated systems produce or optimize content.

### 2.2 AI-generated Advertising Creatives

Advertising creative generation has moved from rule-based templates toward multimodal generation pipelines. Existing studies explore image generation, creative selection, ranking, and click-through-oriented optimization. In online advertising systems, creative quality affects user attention, ad relevance, and downstream ranking performance. However, many industrial studies depend on proprietary datasets and platform-specific CTR signals, limiting reproducibility.

### 2.3 Prompt Optimization and Iterative Generation

Prompt design strongly affects generative AI outputs. Iterative generation improves results by decomposing tasks, refining prompts, and using feedback from prior candidates. This paper extends that logic to digital media advertising by treating each candidate creative as an item in a search space defined by audience, appeal type, layout, tone, CTA strategy, and visual style.

### 2.4 Multi-objective Content Evaluation

Advertising creatives must satisfy multiple objectives simultaneously. A high-quality creative should be relevant to the product, understandable, visually appealing, safe, differentiated from other variants, and persuasive for the intended audience. EAI-CO therefore uses a multi-objective reward rather than optimizing a single metric.

## 3. Methodology

### 3.1 Problem Definition

Given a product `p`, audience segment `a`, platform context `s`, and campaign constraints `c`, the task is to generate a social media advertisement `x = {headline, body, CTA, visual_prompt, image}`. The goal is to select an optimized creative that maximizes a multi-objective reward while satisfying factual and safety constraints.

### 3.2 Campaign Brief Encoder

The campaign brief encoder converts product information into a structured representation:

```json
{
  "product_title": "Portable Espresso Maker",
  "category": "Consumer electronics",
  "selling_points": ["compact", "rechargeable", "travel-friendly"],
  "audience_segment": "young_professionals",
  "platform": "Instagram-style social feed",
  "tone": "energetic",
  "constraints": ["avoid health claims"]
}
```

This representation standardizes inputs for all generation methods and allows controlled comparisons across products and audiences.

### 3.3 Exploratory Creative Generator

The generator creates multiple candidates for each product-audience pair. It varies:

- emotional style: practical, aspirational, playful, premium, warm;
- appeal type: convenience, price value, identity, family benefit, productivity;
- layout style: product-centered, lifestyle scene, comparison layout, minimal poster;
- color direction: high contrast, warm neutral, clean bright, bold accent;
- CTA type: learn more, shop now, try today, compare options;
- caption length: short, medium, detailed;
- audience pain point: time pressure, budget, family needs, study/work focus.

In the full study, a commercial API model generates copy and visual prompts, while an open-source image model generates the visual candidates. The prototype included in this workspace uses deterministic placeholders so that the pipeline can be inspected and extended before connecting external models.

### 3.4 Automatic Multi-objective Evaluator

Each candidate is scored on:

- `relevance`: consistency between product, text, and visual prompt;
- `clarity`: readability and directness of the message;
- `aesthetic`: expected visual quality and layout coherence;
- `audience_fit`: match between creative appeal and audience segment;
- `brand_safety`: absence of unsafe or unsupported claims;
- `factuality_penalty`: penalty for claims not grounded in selling points;
- `diversity`: distance from other candidates in the candidate set;
- `predicted_engagement`: lightweight prediction of user attention and click intention.

The default reward is:

```text
reward = 0.25 relevance
       + 0.20 clarity
       + 0.20 aesthetic
       + 0.20 predicted_engagement
       + 0.15 diversity
       - penalty
```

### 3.5 Exploratory Optimization Loop

EAI-CO uses three rounds:

1. **Exploration round:** generate diverse candidates through controlled sampling.
2. **Refinement round:** select high-scoring and diverse elites, then mutate copy, CTA, and visual prompts.
3. **Selection round:** score final candidates and select the Pareto-preferred or highest-reward creative.

This loop allows quality improvement while avoiding collapse into near-duplicate candidates.

### 3.6 Human-in-the-loop Validation

Human participants evaluate final advertisements only. They do not train or steer the generator. This design preserves separation between generation and validation and allows the study to test whether automatic optimization aligns with human perception.

## 4. Experimental Setup

### 4.1 Data

The full experiment will construct 100 product advertising tasks from public e-commerce or product image datasets. Each product is paired with four audience segments: students, young professionals, family users, and price-sensitive consumers. This yields 400 product-audience tasks.

### 4.2 Compared Methods

- `B0 Template`: rule-based copy template and original product image.
- `B1 Single-shot API`: commercial API model with one generation pass.
- `B2 Open-source only`: open-source LLM and open-source image model.
- `B3 Prompt-engineered AI`: fixed high-quality prompt, no iterative optimization.
- `Ours EAI-CO`: exploratory generation, multi-objective evaluation, and iterative optimization.

### 4.3 Ablation Studies

- `Ours w/o diversity`: remove diversity from reward.
- `Ours w/o factual penalty`: remove factuality and brand-safety penalties.
- `Ours w/o iterative loop`: only use first-round candidates.
- `Ours w/o audience modeling`: remove audience persona from the brief.

### 4.4 Automatic Metrics

The automatic evaluation reports relevance, aesthetic quality, text clarity, audience fit, brand safety, diversity, predicted engagement, total reward, average latency, and estimated model calls per final creative.

### 4.5 Human Evaluation

The human evaluation is intentionally conducted on a stratified subset rather than the full benchmark, which follows common practice for generative-system evaluation under annotation-cost constraints. The default design samples 30 products, 4 audience segments, and 5 primary methods, producing 600 advertisements. Around 20 to 60 participants rate each advertisement on a 7-point Likert scale for attractiveness, information clarity, visual quality, audience relevance, and click/purchase intention, with a target of at least 5 ratings per advertisement in the final submission version and at least 3 in a pilot version. Pairwise preference tests ask participants to choose the advertisement they would be more likely to click under blind evaluation.

### 4.6 Statistical Analysis

Likert ratings will be analyzed using Friedman tests or repeated-measures ANOVA depending on normality assumptions. Pairwise comparisons use Wilcoxon signed-rank tests with Holm-Bonferroni correction. Effect sizes are reported using Cliff's delta or partial eta squared. Spearman correlation tests the relationship between automatic rewards and human-rated click intention. When feasible, the study should also report confidence intervals and inter-rater reliability to strengthen publication-grade methodological transparency.

## 5. Expected Results Structure

The results section should include:

1. A table comparing automatic metrics across methods.
2. A table comparing human ratings across methods.
3. A pairwise preference matrix.
4. An ablation table showing which EAI-CO modules contribute most.
5. A cost-efficiency table reporting model calls and generation time.
6. Example visual cases showing successful optimization and failure cases.

The core hypothesis is that EAI-CO will improve comprehensive reward and human-rated click intention compared with single-shot and fixed-prompt baselines, while maintaining stronger diversity and safety than unconstrained generation.

The full evaluation logic is therefore layered: automatic metrics over the entire benchmark provide breadth, human evaluation over a stratified subset provides perceptual validity, and ablation studies provide causal evidence about the contribution of exploration, diversity preservation, and audience modeling.

## 6. Discussion

The expected advantage of EAI-CO comes from making generation exploratory and measurable. Instead of relying on a single generated answer, the framework searches across creative strategies and selects candidates that balance appeal, clarity, relevance, aesthetics, and diversity. This better reflects real digital media workflows where marketers compare many variants before publication.

The main limitation is that the study does not use real platform CTR data. Therefore, claims should be restricted to predicted engagement and human-rated intention. A second limitation is that automatic evaluators may inherit model biases. Human evaluation and correlation analysis are included to test evaluator validity but cannot fully remove this risk.

An additional reporting constraint is that human evaluation may cover only a subset of the total benchmark. This is acceptable when clearly disclosed and when the subset is stratified, blind-rated, and paired with full-benchmark automatic evaluation.

## 7. Conclusion

This paper presents EAI-CO, an exploratory AI framework for automated social media advertising content creation and multi-objective creative optimization. By integrating structured briefs, diverse generation, automatic evaluation, iterative refinement, and human validation, EAI-CO provides a reproducible method for studying AI-assisted digital media content production. Future work can extend this design to real A/B testing, short-video advertising, multilingual campaigns, and platform-specific creative ranking.

## References to Include

- Grewal, D., Satornino, C. B., Davenport, T., et al. How generative AI is shaping the future of marketing. *Journal of the Academy of Marketing Science*, 2025.
- Heitmann, M. Generative AI for marketing content creation: New rules for an old game. *NIM Marketing Intelligence Review*, 2024.
- Yang, H., Yuan, J., Yang, S., Xu, L., Yuan, S., and Zeng, Y. A new creative generation pipeline for click-through rate with Stable Diffusion model. *WWW Companion*, 2024.
- Lin, K., Zhang, X., Li, F., Wang, P., Long, Q., Deng, H., Xu, J., and Zheng, B. Joint optimization of ad ranking and creative selection. *SIGIR*, 2022.
- Yang, Z., Sang, L., Wang, H., Chen, W., Wang, L., He, J., Peng, C., Lin, Z., Gan, C., and Shao, J. Parallel ranking of ads and creatives in real-time advertising systems. *AAAI*, 2024.
