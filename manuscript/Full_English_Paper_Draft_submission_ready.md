# Exploratory Artificial Intelligence for Automated Social Media Advertising Content Creation and Multi-objective Creative Optimization

## Abstract

Generative artificial intelligence has substantially accelerated digital media content production, yet many operational systems still rely on single-pass generation followed by manual selection. This paradigm limits systematic exploration, weakens reproducibility, and makes optimization difficult to evaluate rigorously. This paper proposes **EAI-CO** (Exploratory AI-based Creative Optimization), a framework that formulates social-media advertising content creation as an exploratory multi-objective optimization problem rather than a one-shot generation task. EAI-CO encodes campaign briefs into structured task representations, generates diverse candidate creatives, scores them with a compliance-aware multi-objective evaluator, and refines promising candidates through iterative search.

We evaluate the framework on a benchmark of 10 products and 4 audience segments, yielding 40 product-audience tasks. The main experiment is conducted on a fully local, reproducible `Qwen2.5-7B-Instruct` backbone and compares EAI-CO against template-based, single-shot, open-source-only, and prompt-engineered baselines, together with four mechanistic ablations. EAI-CO achieves the best overall reward (`0.8051`) and significantly outperforms all primary baselines (`p < 0.001` to `p < 1e-12`). The full method also outperforms all ablated variants, indicating that diversity preservation, iterative refinement, and audience modeling each contribute materially to final creative quality. A supplementary cross-model validation on `gpt-5.4-mini-2026-03-17` preserves the same ranking pattern and confirms transferability of the framework to a stronger commercial generator.

The results support a central claim: for digital-media advertising, the main gain comes not only from stronger generative models, but from treating creative production as an exploratory optimization process with explicit control over audience fit, diversity, and compliance-aware quality.

**Keywords:** exploratory artificial intelligence; generative AI; digital media; social media advertising; automated content creation; creative optimization

## 1. Introduction

Digital-media advertising depends on continuous production of audience-specific creative assets across platforms, products, and campaign stages. In social-media environments, effective advertisements are typically short, visually coherent, audience-aware, and sufficiently differentiated from competing messages. This creates a practical tension: advertisers need both scale and relevance, yet manual creative production remains expensive and slow.

Generative AI provides a plausible solution to this bottleneck, but many current workflows use it in a limited way. A model is given a prompt, one or several outputs are generated, and human operators manually choose among them. This improves throughput, but it does not solve the deeper optimization problem. A one-shot workflow does not systematically explore creative alternatives, offers weak causal insight into why one variant performs better than another, and provides limited support for reproducible evaluation.

This paper argues that automated content creation for digital media should be treated as a structured search problem. In practice, advertisers do not evaluate only one creative at a time; they compare multiple variants differing in tone, appeal type, audience alignment, and visual emphasis. A rigorous AI framework should therefore generate diverse candidates, evaluate them under multiple objectives, preserve beneficial diversity, and iteratively refine the strongest candidates.

To address this need, we propose **EAI-CO** (Exploratory AI-based Creative Optimization), a framework for automated social-media advertising content creation and optimization. EAI-CO integrates five components: (1) structured campaign-brief encoding, (2) exploratory candidate generation, (3) automatic multi-objective evaluation, (4) iterative optimization through refinement and selection, and (5) evaluation protocols designed for reproducibility and scientific inspection. Unlike purely descriptive prompt engineering, EAI-CO operationalizes creative generation as a measurable optimization pipeline.

The contributions of this paper are fourfold:

1. We propose a framework that reformulates social-media ad generation as an exploratory multi-objective optimization problem.
2. We design a compliance-aware evaluator that jointly considers relevance, clarity, aesthetics, audience fit, diversity, safety, and predicted engagement.
3. We provide a reproducible benchmark protocol with primary baselines, mechanistic ablations, audience-wise analyses, and runtime/cost reporting.
4. We demonstrate cross-model transfer by validating the framework on both a local open-source backbone and a stronger commercial model.

## 2. Related Work

### 2.1 Generative AI in Marketing and Advertising

Recent work in marketing and advertising has shown that generative AI can accelerate the production of copy, imagery, and personalized promotional materials. However, marketing applications impose requirements that differ from general-purpose text generation. Advertising creatives must remain on-brand, relevant to the intended audience, visually consistent, and free of unsupported or unsafe claims. Prior discussions of generative AI in marketing therefore emphasize not only productivity gains, but also governance, evaluation, and quality-control challenges.

### 2.2 AI-generated Advertising Creatives

Research on AI-generated advertising creatives has explored creative generation, creative ranking, click-through-oriented selection, and multimodal content pipelines. Some studies focus on image synthesis or CTR-oriented optimization, while others examine the joint selection of ads and creatives within larger serving systems. However, a large part of this literature is built on proprietary data or platform-specific online metrics, which makes scientific comparison and reproducibility difficult.

### 2.3 Prompt Optimization and Iterative Generation

Prompt engineering has become a dominant practical strategy for improving generative outputs. Yet prompt optimization alone does not provide a principled mechanism for maintaining diversity or for explaining how candidate search affects final quality. Iterative generation methods partially address this issue by revising prompts or outputs across steps. Our work extends this direction by embedding iterative refinement inside a structured creative search space with explicit optimization targets.

### 2.4 Multi-objective Evaluation

Advertising quality is inherently multi-objective. A useful creative should be relevant, legible, audience-matched, visually plausible, distinctive from other variants, and compliant with reasonable factual and brand-safety constraints. Accordingly, a single scalar objective such as fluency is inadequate. Our work follows the multi-objective view and treats final quality as the result of balancing several competing factors.

## 3. Methodology

### 3.1 Problem Formulation

Let a campaign task be defined by a product \(p\), an audience segment \(a\), a platform context \(s\), and a set of constraints \(c\). The system must generate a creative
\[
x = \{\text{headline}, \text{body}, \text{cta}, \text{visual\_prompt}\},
\]
with the goal of maximizing a multi-objective reward function while remaining aligned with the product brief and the audience profile.

The optimization target is not a single-shot output, but the best creative selected from an explored candidate set under a structured evaluation process.

### 3.2 Campaign Brief Encoder

The campaign-brief encoder converts each advertising task into a structured representation with the following fields:

- `product_title`
- `category`
- `selling_points`
- `audience_segment`
- `platform`
- `tone`
- `constraints`

For example, a product such as a portable espresso maker can be paired with an audience profile such as `young_professionals`, along with constraints such as avoiding unsupported health claims. This standardization is important for two reasons. First, it ensures fair comparison across methods by keeping the task description fixed. Second, it enables systematic audience-aware generation rather than ad hoc prompting.

### 3.3 Exploratory Candidate Generator

Instead of directly asking the model for a single advertisement, EAI-CO samples a controlled creative space. Each candidate is defined by a combination of exploration axes, including:

- emotional style,
- appeal type,
- layout style,
- color direction,
- CTA style,
- caption length,
- audience pain point.

For each product-audience task, the system generates multiple candidates by varying these axes. Under the final local benchmark configuration, the optimization loop uses two rounds with two candidates per round and retains one elite candidate for refinement.

The output of each candidate includes a headline, a short body, a CTA, and a visual prompt. In the current benchmark implementation, image generation is disabled in order to keep the experiment focused on reproducible text-conditioned creative optimization and to avoid introducing confounds from separate large-scale image synthesis runs. The visual prompt remains part of the output representation because it contributes to multimodal creative coherence.

### 3.4 Multi-objective Evaluator

Each candidate is scored on the following dimensions:

- **Relevance**: alignment between product selling points and the generated creative.
- **Clarity**: compactness and readability of the generated copy.
- **Aesthetic quality**: expected visual coherence and advertising plausibility.
- **Audience fit**: alignment between the creative and the audience priorities/pain points.
- **Brand safety**: avoidance of unsupported or unsafe phrasing.
- **Factuality penalty**: penalty assigned when the copy contains overly aggressive or implausible claims.
- **Diversity**: relative distinctiveness from other candidates in the same task.
- **Predicted engagement**: a lightweight proxy of likely attention/click response.

For the final `Qwen2.5-7B-Instruct` benchmark, the reward weights are:

- relevance: `0.20`
- clarity: `0.16`
- aesthetic: `0.14`
- audience fit: `0.16`
- predicted engagement: `0.16`
- diversity: `0.08`
- brand safety: `0.10`

The reward can therefore be written as:

\[
R(x) = 0.20\,r_{\text{rel}} + 0.16\,r_{\text{clar}} + 0.14\,r_{\text{aes}} + 0.16\,r_{\text{aud}} + 0.16\,r_{\text{eng}} + 0.08\,r_{\text{div}} + 0.10\,r_{\text{safe}} - p_{\text{fact}}.
\]

This evaluator is intentionally compliance-aware. Rather than maximizing persuasive force alone, it rewards a balance between attractiveness and acceptable factual discipline.

### 3.5 Iterative Optimization

EAI-CO performs optimization in three conceptual steps:

1. **Exploration**: generate a diverse initial candidate set by sampling the creative axes.
2. **Refinement**: retain the best-scoring elite candidates and mutate them through controlled variations.
3. **Selection**: rank the resulting candidates using the multi-objective reward and output the strongest creative.

This process makes the generator behave more like a structured optimizer than a direct text synthesizer. The system does not assume that the first answer is best; instead, it treats generation as search under measurable objectives.

### 3.6 Baselines and Ablations

We compare EAI-CO against four primary baselines:

- `B0_Template`: template-based copy generation.
- `B1_SingleShot_API`: one-pass model generation.
- `B2_OpenSource_Only`: open-source-oriented single-pass generation.
- `B3_PromptEngineered_AI`: fixed high-quality prompt without iterative search.

To test the contribution of internal mechanisms, we also evaluate four ablations:

- `w/o diversity`
- `w/o factual penalty`
- `w/o iterative loop`
- `w/o audience modeling`

These ablations are critical because they allow us to move beyond simple leaderboard comparisons and examine which parts of the framework actually drive performance.

## 4. Experimental Setup

### 4.1 Benchmark Design

The main benchmark contains 10 products paired with 4 audience segments:

- students,
- young professionals,
- family users,
- price-sensitive consumers.

This yields 40 product-audience tasks. Under the full local benchmark, 9 final outputs are retained per task: 5 primary methods plus 4 ablation methods, for a total of 360 evaluated creatives.

### 4.2 Models

#### Main local backbone

The main benchmark is conducted on a local `Qwen2.5-7B-Instruct` backbone running on an RTX 3090 GPU. This setup is chosen because it is reproducible, locally controllable, and strong enough to serve as a realistic open-source generation backbone.

#### Commercial cross-model validation

To test transferability beyond the local backbone, we perform a supplementary validation on `gpt-5.4-mini-2026-03-17`. This experiment uses the same 10 products and 4 audience segments but evaluates only the 5 primary methods. The purpose is not to replace the main benchmark, but to test whether the ranking advantage of EAI-CO survives under a stronger commercial model.

### 4.3 Runtime Configuration

For the final local main experiment:

- rounds: `2`
- candidates per round: `2`
- elite count: `1`

For the commercial cross-model validation, the same search depth is retained to keep the protocol consistent across model families.

### 4.4 Evaluation and Statistical Testing

All reported results are computed using the same automatic evaluator and the same task protocol within each experiment. Statistical testing compares `Ours_EAI_CO` against each primary baseline using paired comparisons over the shared task set. We report:

- reward differences,
- significance levels,
- predicted-engagement differences,
- audience-fit differences.

We further report audience-wise breakdowns and per-method runtime/cost statistics to improve interpretability and reproducibility.

### 4.5 Reproducibility

To support verification, the project includes:

- executable scripts,
- fixed experiment configurations,
- local-output CSV files,
- benchmark summaries,
- analysis scripts for method-level tables and significance tests.

The main repository excludes large result directories from version control, but the pipeline itself is fully documented and executable.

## 5. Results

### 5.1 Main Benchmark Results on `Qwen2.5-7B-Instruct`

The proposed EAI-CO framework achieved the highest overall reward among all primary baselines. The mean reward values were:

- `Ours_EAI_CO = 0.8051`
- `B3_PromptEngineered_AI = 0.7568`
- `B1_SingleShot_API = 0.7494`
- `B2_OpenSource_Only = 0.7445`
- `B0_Template = 0.6856`

The gains over every baseline were statistically significant. Relative to `B0_Template`, EAI-CO improved reward by `+0.1195` (`p = 9.09e-13`). It also improved reward by `+0.0557` over `B1_SingleShot_API` (`p = 1.34e-06`), `+0.0606` over `B2_OpenSource_Only` (`p = 5.18e-08`), and `+0.0483` over `B3_PromptEngineered_AI` (`p = 6.58e-05`).

Beyond the scalar reward, the full framework achieved the best audience fit among the primary methods (`0.4719`) and the strongest brand-safety score (`0.9945`). These patterns indicate that the advantage is not restricted to one metric dimension such as relevance or engagement, but reflects a more balanced optimization outcome.

### 5.2 Ablation Results

The ablation study further supports the mechanism of the framework. The mean reward values were:

- `EAI-CO = 0.8051`
- `w/o factual penalty = 0.8006`
- `w/o audience modeling = 0.7833`
- `w/o iterative loop = 0.7760`
- `w/o diversity = 0.7349`

Removing diversity produced the largest drop (`-0.0702`), suggesting that search-space breadth is a major determinant of the final gain. Removing iterative refinement reduced reward by `-0.0291`, confirming that multi-step optimization adds value beyond first-pass generation. Removing audience modeling lowered reward by `-0.0218` and decreased audience fit from `0.4719` to `0.3960`, which provides direct evidence that persona conditioning has measurable functional value.

The factual-penalty ablation remained close to the full framework (`-0.0045`). This should not be read as a failure of the compliance mechanism. Rather, it indicates that compliance-aware constraints can be enforced with only a small tradeoff in raw persuasive reward, which is desirable in practical advertising settings.

### 5.3 Audience-wise Analysis

EAI-CO ranked first in all four audience groups:

- `family_users = 0.8147`
- `students = 0.8098`
- `young_professionals = 0.8050`
- `price_sensitive_consumers = 0.7908`

This pattern reduces the risk that the overall result is driven by a single particularly easy subgroup. It also suggests that the exploratory optimization strategy adapts robustly to different audience priorities.

### 5.4 Runtime and Cost

The average per-task latency values were:

- `Ours_EAI_CO = 11114 ms`
- `B3_PromptEngineered_AI = 2727 ms`
- `B1_SingleShot_API = 2780 ms`
- `B2_OpenSource_Only = 2707 ms`
- `B0_Template = 1 ms`

The full framework required `8.0` mean model calls per task, compared with `4.0`, `4.0`, and `3.0` for the three single-pass baselines. The increased cost is expected, because EAI-CO explicitly trades computation for search quality. In an offline campaign-design scenario, this tradeoff is acceptable.

### 5.5 Cross-model Validation on `gpt-5.4-mini-2026-03-17`

The cross-model validation preserved the same ranking pattern observed on the local open-source backbone. The mean reward values were:

- `Ours_EAI_CO = 0.9957`
- `B2_OpenSource_Only = 0.9751`
- `B1_SingleShot_API = 0.9736`
- `B3_PromptEngineered_AI = 0.9729`
- `B0_Template = 0.8396`

Again, all gains were statistically significant. Relative to `B0_Template`, EAI-CO improved reward by `+0.1561` (`p = 1.54e-08`). The gains over the stronger single-pass baselines remained positive and significant: `+0.0221` over `B1_SingleShot_API` (`p = 1.41e-03`), `+0.0206` over `B2_OpenSource_Only` (`p = 6.42e-04`), and `+0.0228` over `B3_PromptEngineered_AI` (`p = 1.01e-03`).

Audience-wise, EAI-CO remained first for all four groups:

- `family_users = 0.9950`
- `price_sensitive_consumers = 0.9938`
- `students = 1.0000`
- `young_professionals = 0.9939`

These results show that the ranking advantage of EAI-CO transfers to a stronger commercial model under a matched protocol.

## 6. Discussion

### 6.1 Exploratory Optimization Outperforms One-shot Generation

The most consistent result across experiments is that exploratory optimization outperforms one-shot generation. The framework was superior on the local `Qwen2.5-7B-Instruct` backbone and preserved the same superiority ordering on `gpt-5.4-mini`. This strongly suggests that the main source of benefit lies in the optimization structure rather than in any single model family.

The empirical implication is important: in digital-media advertising, creative quality should not be conceptualized purely as a text-generation problem. It should be treated as a search problem involving candidate generation, comparison, refinement, and selection under multiple objectives.

### 6.2 Why Diversity and Iteration Matter

The ablation study indicates that diversity preservation is the strongest individual contributor to performance, followed by iterative refinement. This is consistent with how creative work operates in practice. Campaign quality rarely emerges from a single draft; it emerges from comparing alternatives that frame the same product differently. By preserving structural variation among candidates, EAI-CO avoids premature convergence and improves the odds of discovering a high-performing creative strategy.

### 6.3 Audience Modeling Has Measurable Value

The audience-modeling ablation is especially useful because it moves the discussion of personalization from rhetoric to evidence. Once persona conditioning is removed, both reward and audience-fit quality decline. This means that the framework is not merely generating generic persuasive copy; it is measurably exploiting audience-specific information.

### 6.4 Compliance-aware Optimization Is a Practical Requirement

The factual-penalty ablation remained close to the full model in the main benchmark, which indicates that compliance-aware scoring can be integrated without substantially damaging persuasive performance. This is a practically desirable result. In real digital-media systems, a method that achieves marginally higher persuasive force by encouraging unsupported claims would not necessarily be preferable. The present results therefore support a more deployment-relevant interpretation of optimization quality.

### 6.5 Cross-model Transfer Improves the Strength of the Paper

The commercial-model experiment is not intended to replace the local main benchmark. Its function is different: it tests whether the advantage of EAI-CO survives when the backbone changes to a stronger commercial model. The answer is yes. This improves the credibility of the paper by reducing the risk that the main result is merely an artifact of one local generator.

At the same time, the `gpt-5.4-mini` results exhibit ceiling compression, with reward values clustered close to 1.0. This means the commercial-model validation is best interpreted as evidence of transferability and ranking preservation, not as the main source of effect-size contrast.

### 6.6 Limitations

This study has several limitations.

First, it does not use real platform CTR or conversion outcomes. Accordingly, the paper should restrict its claims to automatic reward, predicted engagement, audience-conditioned quality, and cross-model ranking consistency.

Second, the current submission package does not include a completed human evaluation study. This does not invalidate the automatic and cross-model findings, but it does narrow the set of claims that can be made. The current evidence supports framework-level superiority under the proposed evaluator; it does not yet support verified human preference gains.

Third, the present benchmark size is scientifically useful and sufficient for method comparison, but still smaller than a full-scale industrial ad-production environment. Future work should expand both product diversity and real-world deployment conditions.

## 7. Conclusion

This paper presented EAI-CO, a framework that treats social-media advertising content creation as an exploratory multi-objective optimization problem. Across a reproducible local benchmark on `Qwen2.5-7B-Instruct`, the framework significantly outperformed template-based and single-pass baselines and remained superior under a commercial cross-model validation on `gpt-5.4-mini-2026-03-17`. The ablation study showed that diversity preservation, iterative refinement, and audience modeling each contribute to the observed gains.

The broader implication is that the core scientific contribution of AI-assisted advertising does not lie solely in stronger generative backbones. It lies in building optimization frameworks that explicitly search, evaluate, and refine creative alternatives under measurable constraints. This provides a more rigorous foundation for future work on digital-media content generation, audience-aware creative design, and responsible automated marketing systems.

## Reproducibility and Availability Statement

The experiments are implemented as script-driven pipelines with explicit configuration files, model backbones, output CSV files, and deterministic benchmark definitions. The local project includes the code required to reproduce the main benchmark and the cross-model validation under the reported settings. Large generated output directories are excluded from version control for repository hygiene, but the workflow and analysis scripts are provided.

## Ethics and Scope Statement

The present version of the study evaluates generated advertising creatives using automatic and proxy metrics only and does not claim validated real-world click-through-rate improvement. The framework is intended for research on creative optimization and should not be interpreted as a substitute for human review in regulated or high-risk advertising contexts.
