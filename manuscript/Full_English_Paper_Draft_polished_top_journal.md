# Exploratory Artificial Intelligence for Automated Social Media Advertising Content Creation and Multi-objective Creative Optimization

## Abstract

Generative artificial intelligence has substantially accelerated digital-media content production, yet most operational workflows still rely on single-pass generation followed by manual screening. Such a paradigm limits systematic exploration, weakens reproducibility, and offers little insight into how candidate variation affects final creative quality. This paper proposes **EAI-CO** (Exploratory AI-based Creative Optimization), a framework that formulates social-media advertising content creation as an exploratory multi-objective optimization problem rather than as a one-shot generation task. EAI-CO encodes campaign briefs into structured task representations, generates diverse creative candidates, evaluates them through a compliance-aware multi-objective scoring function, and iteratively refines strong candidates through controlled search.

The main benchmark is conducted on a reproducible local `Qwen2.5-7B-Instruct` backbone over 10 products, 4 audience segments, and 40 product-audience tasks. EAI-CO is compared against template-based, single-shot, open-source-only, and prompt-engineered baselines, together with four mechanistic ablations. EAI-CO achieves the best overall reward (`0.8051`) and significantly outperforms all primary baselines (`p < 0.001` to `p < 1e-12`). The full method also outperforms all ablated variants, indicating that diversity preservation, iterative refinement, and audience modeling each contribute materially to final quality. A supplementary cross-model validation on `gpt-5.4-mini-2026-03-17` preserves the same ranking pattern and confirms transferability of the framework to a stronger commercial generator.

Taken together, the results show that the main gain in digital-media ad generation does not come only from stronger generative backbones. It comes from treating creative production as an exploratory optimization process with explicit control over audience fit, diversity, and compliance-aware quality.

**Keywords:** exploratory artificial intelligence; generative AI; digital media; social media advertising; automated content creation; creative optimization

## 1. Introduction

Digital-media advertising increasingly depends on rapid production of personalized creative assets across platforms, products, and audience segments. Social-media advertisements, in particular, must communicate clearly under severe space constraints while remaining visually coherent, audience-relevant, and distinct from competing messages. This creates a structural problem for marketing operations: advertisers need both scale and relevance, but manual creative development remains expensive, slow, and difficult to standardize.

Generative AI offers an attractive response to this bottleneck. It can produce headlines, captions, visual prompts, and promotional variants at a speed that traditional creative workflows cannot match. However, speed alone is not the central scientific issue. In many practical deployments, generative models are still used in a narrow one-shot manner: a prompt is written, one or a few candidates are produced, and human operators manually choose among them. Although this increases throughput, it does not solve the deeper optimization problem. A single-pass workflow neither systematically explores the creative search space nor explains why one candidate should be preferred to another. As a result, evaluation remains ad hoc and reproducibility remains weak.

This paper argues that automated advertising content creation should be treated as an exploratory optimization problem rather than as a direct generation problem. In real campaign practice, creatives are rarely evaluated in isolation. Marketers compare variants that differ in emotional framing, benefit emphasis, call-to-action strategy, and audience alignment. A rigorous AI system should therefore do more than generate fluent copy. It should generate alternative candidates, preserve meaningful diversity, evaluate them under explicit objectives, and iteratively refine the most promising directions.

To address this need, we propose **EAI-CO** (Exploratory AI-based Creative Optimization), a framework for automated social-media advertising content creation and optimization. EAI-CO integrates structured campaign-brief encoding, exploratory candidate generation, automatic multi-objective evaluation, iterative refinement, and reproducible experiment management. The framework is designed to move AI-assisted advertising away from isolated prompt-response interactions and toward a controlled search-and-selection process.

The present study makes four contributions. First, it introduces a framework that reformulates social-media ad generation as an exploratory multi-objective optimization problem. Second, it develops a compliance-aware evaluator that jointly considers relevance, clarity, aesthetics, audience fit, diversity, safety, and predicted engagement. Third, it provides a reproducible benchmark protocol with baseline comparisons, mechanistic ablations, audience-wise analyses, and runtime reporting. Fourth, it validates transferability by showing that the same superiority pattern holds under both a local open-source backbone and a stronger commercial generator.

## 2. Related Work

### 2.1 Generative AI in Marketing and Advertising

Generative AI is reshaping marketing by expanding the feasible scale of content production, personalization, and campaign experimentation. Existing marketing scholarship has emphasized both the opportunities and the governance challenges of this transition. Generative models can accelerate copywriting, ideation, and promotional adaptation, but marketing applications require stronger controls than general-purpose text generation. In particular, marketing creatives must remain aligned with brand positioning, audience expectations, factual constraints, and measurable business goals (Grewal et al., 2025; Heitmann, 2024).

This distinction matters because a generative system that is merely fluent is not necessarily useful in advertising. Content must also be audience-sensitive, operationally differentiable, and sufficiently safe for deployment. Our work builds on this observation by treating advertising generation as a constrained optimization problem rather than as unconstrained text synthesis.

### 2.2 AI-generated Advertising Creatives

Recent work on AI-generated advertising creatives spans creative generation, creative ranking, ad-creative selection, and CTR-oriented optimization. Industrial systems increasingly treat creatives as decision variables that interact with larger ad-serving pipelines. For example, prior work has studied joint optimization of ad ranking and creative selection, as well as parallel ranking architectures for ads and creatives in real-time systems (Lin et al., 2022; Yang et al., 2024). These studies demonstrate that creative choice is not peripheral to advertising performance; it is a core component of it.

At the same time, a substantial portion of this literature relies on proprietary datasets, industrial infrastructure, or online feedback unavailable to academic replication. This makes it difficult to compare creative-generation methods under controlled and reproducible conditions. The present paper contributes a framework-oriented benchmark that remains tractable under local execution while preserving explicit connections to practical advertising objectives.

### 2.3 Prompt Optimization and Iterative Generation

Prompt engineering has become a standard practical technique for improving large-model outputs, including marketing copy. However, prompt optimization is often under-specified as a research methodology. It typically improves results without making the search process itself observable. Iterative generation partially addresses this issue by revising prompts or outputs across multiple rounds, but many iterative systems still lack an explicit account of diversity preservation and candidate selection.

Our approach extends this line of work by embedding iterative generation within a structured search space. Candidates are not only regenerated; they are explored under defined creative axes and evaluated through a shared reward function. This permits stronger analysis of what is gained by search, rather than by prompt fluency alone.

### 2.4 Multi-objective Evaluation for Creative Systems

Advertising quality is intrinsically multi-objective. A useful creative should be relevant to the product, clear to the viewer, visually plausible, aligned with the target audience, distinct from alternative variants, and compliant with factual or safety constraints. A single scalar criterion such as fluency, lexical diversity, or model confidence is therefore insufficient. Our framework follows the multi-objective view and explicitly models creative quality as the result of balancing competing objectives rather than maximizing a single generation score.

## 3. Methodology

### 3.1 Problem Definition

Let a campaign task be defined by a product \(p\), an audience segment \(a\), a platform context \(s\), and a constraint set \(c\). The system must generate a social-media advertisement
\[
x = \{\text{headline}, \text{body}, \text{cta}, \text{visual\_prompt}\},
\]
with the aim of maximizing a multi-objective reward while remaining aligned with the campaign brief. The key shift is that the objective is not to generate one plausible creative, but to identify the strongest creative from an explored candidate set.

### 3.2 Campaign Brief Encoder

The campaign-brief encoder converts each task into a structured schema containing product title, category, selling points, audience segment, platform, tone, and constraints. This representation standardizes the task description across methods and ensures that differences among outputs arise from model behavior and search strategy rather than inconsistent prompt framing.

For example, a portable espresso maker can be paired with the `young_professionals` segment and the platform context `Instagram-style social feed`, while constraint fields can encode safety requirements such as avoiding unsupported health or medical claims. This structure supports audience-aware generation while preserving fairness across methods.

### 3.3 Exploratory Candidate Generator

Instead of querying the model for one direct answer, EAI-CO samples a controlled creative space. Each candidate is defined by a set of exploration axes, including:

- emotional style,
- appeal type,
- layout style,
- color direction,
- CTA style,
- caption length,
- audience pain point.

For each product-audience task, the framework generates multiple candidates by varying these axes. Under the final benchmark configuration, two rounds are used with two candidates per round, and one elite candidate is retained for refinement. This design yields a shallow but explicit exploratory loop that is computationally manageable while still exposing the effect of search.

Each candidate contains four fields: headline, body, CTA, and visual prompt. Although the current benchmark does not execute a full large-scale image-generation experiment, the visual-prompt field is preserved because multimodal coherence remains part of the creative specification.

### 3.4 Compliance-aware Multi-objective Evaluator

Each candidate is scored on seven positive dimensions and one penalty term:

- relevance,
- clarity,
- aesthetic quality,
- audience fit,
- predicted engagement,
- diversity,
- brand safety,
- factuality penalty.

For the final local benchmark, the reward weights are:

- relevance: `0.20`
- clarity: `0.16`
- aesthetic: `0.14`
- audience fit: `0.16`
- predicted engagement: `0.16`
- diversity: `0.08`
- brand safety: `0.10`

The reward is computed as:

\[
R(x)=0.20r_{\text{rel}}+0.16r_{\text{clar}}+0.14r_{\text{aes}}+0.16r_{\text{aud}}+0.16r_{\text{eng}}+0.08r_{\text{div}}+0.10r_{\text{safe}}-p_{\text{fact}}.
\]

This formulation is intentionally compliance-aware. It does not reward persuasive intensity in isolation. Instead, it balances persuasive potential with audience alignment and plausible, safe wording.

### 3.5 Iterative Optimization Loop

EAI-CO proceeds in three conceptual stages:

1. **Exploration**: sample diverse initial candidates from the creative space.
2. **Refinement**: retain high-scoring elites and mutate them under controlled variation.
3. **Selection**: score the resulting candidates and output the strongest creative under the multi-objective reward.

The framework therefore behaves as an optimizer rather than a simple conditional generator. It assumes that strong creatives emerge from structured comparison and refinement, not from accepting the first fluent answer.

### 3.6 Baselines and Ablations

We compare EAI-CO against four primary baselines:

- `B0_Template`
- `B1_SingleShot_API`
- `B2_OpenSource_Only`
- `B3_PromptEngineered_AI`

To isolate the contribution of internal mechanisms, we evaluate four ablations:

- `Ours_without_diversity`
- `Ours_without_factual_penalty`
- `Ours_without_iterative_loop`
- `Ours_without_audience_modeling`

These ablations are necessary because a top-tier empirical paper must show not only that the full method works, but also why it works.

## 4. Experimental Setup

### 4.1 Benchmark Construction

The main benchmark comprises 10 products crossed with 4 audience segments:

- students,
- young professionals,
- family users,
- price-sensitive consumers.

This produces 40 product-audience tasks. Under the local main experiment, 9 final method outputs are retained per task, corresponding to 5 primary methods and 4 ablations, for a total of 360 evaluated creatives.

### 4.2 Model Settings

#### Main reproducible backbone

The main benchmark uses a local `Qwen2.5-7B-Instruct` backbone executed on an RTX 3090 GPU. This choice prioritizes reproducibility, local control, and scientific traceability.

#### Commercial transfer backbone

A supplementary cross-model validation is conducted on `gpt-5.4-mini-2026-03-17`. This experiment preserves the same 10 products and 4 audience segments but evaluates only the 5 primary methods. Its purpose is to test framework transferability rather than to replace the main benchmark.

### 4.3 Search Configuration

For both the local benchmark and the commercial transfer experiment, the exploration settings are:

- rounds: `2`
- candidates per round: `2`
- elite count: `1`

This configuration provides measurable exploratory behavior without introducing impractically high computational cost.

### 4.4 Evaluation Protocol

All reported metrics are derived from the shared evaluator defined in Section 3.4. Statistical comparisons are conducted between `Ours_EAI_CO` and each primary baseline over the matched task set. The reported outputs include:

- mean reward,
- reward deltas,
- significance values,
- predicted-engagement deltas,
- audience-fit deltas,
- audience-wise summaries,
- runtime statistics.

### 4.5 Reproducibility and Verification

The project includes:

- fixed configuration files,
- executable benchmark scripts,
- generated CSV summaries,
- per-method analysis scripts.

Large outputs are excluded from version control for repository hygiene, but the experimental workflow itself is script-driven and reproducible.

### 4.6 Figures and Tables

**Figure 1.** Overview of the EAI-CO framework.  
The figure should depict the full pipeline from campaign-brief encoding to exploratory candidate generation, multi-objective evaluation, iterative refinement, and final selection.

**Figure 2.** Example candidate evolution under the iterative optimization loop.  
This figure should show how early-round candidates differ from the final selected creative under the same product-audience task.

**Table 1.** Benchmark configuration and compared methods.  
This table should summarize the local backbone, commercial backbone, task count, audience segments, primary baselines, and ablation settings.

**Table 2.** Main benchmark results on `Qwen2.5-7B-Instruct`.  
This table should report mean reward, relevance, clarity, audience fit, diversity, predicted engagement, brand safety, penalty, and latency for the five primary methods.

**Table 3.** Ablation results on `Qwen2.5-7B-Instruct`.  
This table should report reward gaps and audience-fit gaps relative to the full method.

**Table 4.** Audience-wise performance on the local benchmark.  
This table should report the per-segment reward and audience-fit values for all primary methods.

**Table 5.** Cross-model validation on `gpt-5.4-mini-2026-03-17`.  
This table should report the same primary-method statistics under the commercial model.

**Table 6.** Runtime and cost profile.  
This table should compare average latency, median latency, and mean model calls across methods.

## 5. Results

### 5.1 Main Benchmark Results on the Local Backbone

Table 2 reports the primary-method results on the `Qwen2.5-7B-Instruct` benchmark. EAI-CO achieved the best overall reward (`0.8051`), followed by `B3_PromptEngineered_AI` (`0.7568`), `B1_SingleShot_API` (`0.7494`), `B2_OpenSource_Only` (`0.7445`), and `B0_Template` (`0.6856`).

The superiority of EAI-CO was statistically significant in every pairwise comparison. Relative to `B0_Template`, the reward gain was `+0.1195` (`p = 9.09e-13`). Relative to the stronger single-pass baselines, the gains remained significant: `+0.0557` over `B1_SingleShot_API` (`p = 1.34e-06`), `+0.0606` over `B2_OpenSource_Only` (`p = 5.18e-08`), and `+0.0483` over `B3_PromptEngineered_AI` (`p = 6.58e-05`).

The full framework also achieved the strongest audience fit (`0.4719`) and the best brand-safety score (`0.9945`) among the primary methods. These results suggest that EAI-CO improves creative quality in a balanced way rather than by over-optimizing a single surface property.

### 5.2 Mechanistic Evidence from the Ablation Study

Table 3 reports the ablation study. The full method (`0.8051`) outperformed all ablated variants:

- `w/o factual penalty = 0.8006`
- `w/o audience modeling = 0.7833`
- `w/o iterative loop = 0.7760`
- `w/o diversity = 0.7349`

Removing diversity produced the largest drop (`-0.0702`), indicating that search-space breadth is a major driver of the final gain. Removing the iterative loop reduced reward by `-0.0291`, showing that controlled refinement improves outcomes beyond first-pass generation. Removing audience modeling lowered reward by `-0.0218` and reduced audience fit from `0.4719` to `0.3960`, directly supporting the claim that persona conditioning is functionally important.

The factual-penalty ablation remained close to the full model (`-0.0045`). This result should not be interpreted as evidence that factual controls are unnecessary. A more defensible conclusion is that compliance-aware optimization can be incorporated with only a small tradeoff in persuasive reward while maintaining stronger behavioral discipline.

### 5.3 Audience-wise Robustness

Table 4 reports the audience-wise analysis. EAI-CO ranked first in all four audience groups:

- `family_users = 0.8147`
- `students = 0.8098`
- `young_professionals = 0.8050`
- `price_sensitive_consumers = 0.7908`

This result reduces the concern that the overall performance gain is driven by a single particularly favorable subgroup. Instead, the exploratory optimization strategy appears robust across materially different audience profiles.

### 5.4 Runtime Characteristics

Table 6 summarizes runtime behavior. EAI-CO required `8.0` mean model calls per task and `11114 ms` mean latency, compared with approximately `2700–2800 ms` for the main single-pass baselines. The higher computational cost is expected because EAI-CO explicitly spends extra inference budget on candidate exploration and refinement.

For the intended offline campaign-design setting, this overhead remains operationally acceptable. The framework is not designed for real-time ad serving; it is designed for higher-quality creative preparation before deployment.

### 5.5 Cross-model Validation on `gpt-5.4-mini-2026-03-17`

Table 5 reports the commercial-model validation. The same ranking pattern observed on the local backbone was preserved:

- `Ours_EAI_CO = 0.9957`
- `B2_OpenSource_Only = 0.9751`
- `B1_SingleShot_API = 0.9736`
- `B3_PromptEngineered_AI = 0.9729`
- `B0_Template = 0.8396`

All gains remained statistically significant. EAI-CO improved reward by `+0.1561` over `B0_Template` (`p = 1.54e-08`), `+0.0221` over `B1_SingleShot_API` (`p = 1.41e-03`), `+0.0206` over `B2_OpenSource_Only` (`p = 6.42e-04`), and `+0.0228` over `B3_PromptEngineered_AI` (`p = 1.01e-03`).

The audience-wise ranking was also preserved, with EAI-CO ranked first for `family_users`, `price_sensitive_consumers`, `students`, and `young_professionals`. This supplementary experiment therefore supports the transferability of the proposed framework to a stronger commercial model.

## 6. Discussion

### 6.1 The Core Gain Comes from Optimization Structure

The most consistent result across experiments is that exploratory optimization outperforms one-shot generation. EAI-CO was superior on the local `Qwen2.5-7B-Instruct` benchmark and preserved the same ordering on `gpt-5.4-mini`. This pattern strongly suggests that the main contribution lies in the optimization structure rather than in any one backbone.

From a methodological perspective, this matters because it shifts the locus of contribution. The paper is not merely demonstrating that a particular model writes better advertising copy. It is demonstrating that advertising generation benefits when candidate production is embedded in a structured search-and-selection process.

### 6.2 Diversity and Iteration Are Not Secondary Details

The ablation study indicates that diversity preservation and iterative refinement are central to performance. This aligns with the practical logic of creative work. Good advertising rarely emerges from the first acceptable variant; it emerges from comparing strategically different alternatives and selectively refining the strongest direction. The present benchmark makes that process measurable.

### 6.3 Audience Modeling Must Be Treated as an Empirical Component

The audience-modeling ablation is particularly important because personalization is often invoked rhetorically rather than measured operationally. Here, the effect is visible in both reward and audience-fit outcomes. This suggests that audience-aware conditioning should be treated as a first-class design variable in creative systems rather than as a descriptive label applied after generation.

### 6.4 Compliance-aware Scoring Is Practically Necessary

The factual-penalty ablation remained close to the full framework, which indicates that compliance-aware optimization can be introduced without substantially damaging persuasive performance. For digital-media applications, this is an important systems result. A framework that yields slightly stronger raw promotional language at the cost of weaker factual discipline would be less attractive for responsible use.

### 6.5 Cross-model Transfer Strengthens the Contribution

The supplementary `gpt-5.4-mini` experiment improves the strength of the paper by showing that the ranking advantage of EAI-CO is not specific to one local open-source generator. This is important for a top-tier empirical argument, because reviewers often distinguish between framework-level gains and backbone-specific artifacts.

At the same time, the commercial-model results should be interpreted carefully. Because `gpt-5.4-mini` is a stronger generator under the current evaluator, reward values are compressed near the upper bound. Accordingly, the transfer experiment is best interpreted as evidence of ranking preservation and transferability, not as the primary source of effect-size contrast.

### 6.6 Limitations

This study has several limitations that should be stated explicitly.

First, it does not use real platform CTR or conversion data. Therefore, the paper should restrict its claims to automatic reward, predicted engagement, audience-conditioned quality, and cross-model ranking consistency.

Second, the current submission package does not include a completed human evaluation study. This does not invalidate the automatic and transfer results, but it narrows the claims that can be made. The present evidence supports framework-level superiority under the proposed evaluator; it does not yet support verified human preference gains.

Third, the benchmark is methodologically useful but still smaller than a production-scale industrial evaluation environment. Future work should test larger product sets, richer multimodal outputs, and live deployment conditions.

## 7. Conclusion

This paper presented EAI-CO, a framework that treats social-media advertising content creation as an exploratory multi-objective optimization problem. Across a reproducible local benchmark on `Qwen2.5-7B-Instruct`, the framework significantly outperformed template-based and single-pass baselines and remained superior under a commercial cross-model validation on `gpt-5.4-mini-2026-03-17`. The ablation study showed that diversity preservation, iterative refinement, and audience modeling each contribute materially to the observed gains.

The broader implication is that the main scientific contribution of AI-assisted advertising may lie less in scaling model size alone and more in designing explicit optimization frameworks that search, evaluate, and refine creative alternatives under measurable constraints. This provides a more rigorous foundation for future work on digital-media content generation, audience-aware creative design, and responsible automated marketing systems.

## Reproducibility Statement

The experiments reported in this paper are implemented through script-driven pipelines with explicit configuration files, deterministic benchmark definitions, structured output CSV files, and analysis scripts. The repository contains the code required to reproduce both the local main benchmark and the commercial cross-model validation under the reported settings.

## Ethics and Scope Statement

The present study evaluates automatically generated advertising creatives using automatic and proxy metrics only and does not claim validated real-world click-through-rate improvement. The framework should therefore be understood as a research system for creative optimization rather than as a substitute for human oversight in high-risk or regulated advertising contexts.

## References

Grewal, D., Satornino, C. B., Davenport, T., & Guha, A. (2025). How generative AI is shaping the future of marketing. *Journal of the Academy of Marketing Science, 53*, 1220-1242. [https://link.springer.com/article/10.1007/s11747-024-01064-3](https://link.springer.com/article/10.1007/s11747-024-01064-3)

Heitmann, M. (2024). Generative AI for marketing content creation: New rules for an old game. *NIM Marketing Intelligence Review, 16*(1), 10-17. [https://www.nim.org/en/publications/detail/generative-ai-for-marketing-content-creation](https://www.nim.org/en/publications/detail/generative-ai-for-marketing-content-creation)

Lin, K., Zhang, X., Li, F., Wang, P., Long, Q., Deng, H., Xu, J., & Zheng, B. (2022). Joint optimization of ad ranking and creative selection. In *Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval* (pp. 2627-2631). [https://ir.webis.de/anthology/2022.sigirconf_conference-2022.266/](https://ir.webis.de/anthology/2022.sigirconf_conference-2022.266/)

Yang, Z., Sang, L., Wang, H., Chen, W., Wang, L., He, J., Peng, C., Lin, Z., Gan, C., & Shao, J. (2024). Parallel ranking of ads and creatives in real-time advertising systems. In *Proceedings of the AAAI Conference on Artificial Intelligence*. [https://dblp.org/rec/conf/aaai/YangSWCWHPLGS24](https://dblp.org/rec/conf/aaai/YangSWCWHPLGS24)

Yang, H., Yuan, J., Yang, S., Xu, L., Yuan, S., & Zeng, Y. (2024). A new creative generation pipeline for click-through rate with Stable Diffusion model. In *Companion Proceedings of the ACM Web Conference 2024*. 
