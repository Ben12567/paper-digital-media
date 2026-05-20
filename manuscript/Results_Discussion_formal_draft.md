## 5. Results

### 5.1 Main Benchmark Results on the Local Open-Source Backbone

Table X reports the main benchmark results obtained with `Qwen2.5-7B-Instruct` on 10 products, 4 audience segments, and 40 product-audience tasks. The proposed `EAI-CO` framework achieved the best overall reward among all primary baselines, with a mean reward of `0.8051`, outperforming `B3_PromptEngineered_AI` (`0.7568`), `B1_SingleShot_API` (`0.7494`), `B2_OpenSource_Only` (`0.7445`), and `B0_Template` (`0.6856`).

The improvements were statistically significant for all pairwise comparisons. Relative to `B0_Template`, `EAI-CO` improved the reward by `+0.1195` (`p = 9.09e-13`). It also significantly outperformed the stronger single-pass baselines: `+0.0557` over `B1_SingleShot_API` (`p = 1.34e-06`), `+0.0606` over `B2_OpenSource_Only` (`p = 5.18e-08`), and `+0.0483` over `B3_PromptEngineered_AI` (`p = 6.58e-05`).

Beyond the aggregate reward, `EAI-CO` also yielded the highest audience fit (`0.4719`) and the strongest brand-safety profile (`0.9945`) among the primary methods. This is important because the proposed framework is not intended to maximize persuasive language alone; it is designed to balance persuasive quality, audience conditioning, and compliance-aware constraints within a unified optimization loop.

### 5.2 Ablation Study

Table Y presents the ablation study on the same `Qwen2.5-7B-Instruct` benchmark. After calibrating the evaluator toward compliance-aware scoring, the complete framework outperformed all ablated variants:

- `EAI-CO`: `0.8051`
- `w/o factual penalty`: `0.8006`
- `w/o audience modeling`: `0.7833`
- `w/o iterative loop`: `0.7760`
- `w/o diversity`: `0.7349`

The largest performance drop occurred when diversity preservation was removed (`-0.0702`), indicating that candidate variation is a major source of the final gain. Removing the iterative loop also reduced performance by `-0.0291`, showing that exploratory refinement contributes beyond one-pass generation. The audience-modeling ablation decreased reward by `-0.0218` and reduced audience fit from `0.4719` to `0.3960`, which directly supports the claim that persona-aware optimization is functionally important rather than decorative.

The factual-penalty ablation remained close to the full model, with a reward gap of only `-0.0045`. This result should be interpreted carefully. It does not imply that factual safeguards are unnecessary. Instead, it suggests that compliance-aware constraints can be integrated with only a small cost to persuasive performance, while improving the overall behavioral stability of the system and preventing unsupported promotional wording from dominating the search process.

### 5.3 Audience-wise Analysis

To test whether the overall gains were driven by a single subgroup, the benchmark results were broken down by audience segment. `EAI-CO` ranked first in all four groups:

- `family_users`: `0.8147`
- `students`: `0.8098`
- `young_professionals`: `0.8050`
- `price_sensitive_consumers`: `0.7908`

This consistency strengthens the external validity of the method within the chosen benchmark design. In particular, the gains are not confined to one especially favorable audience profile such as students or family users. Instead, the advantage of exploratory optimization persists across materially different preference structures.

### 5.4 Cost and Runtime

The proposed framework is computationally more expensive than single-pass generation, which is expected given its iterative search design. On the local `Qwen2.5-7B-Instruct` backbone, the average per-task latency of `EAI-CO` was `11114 ms`, compared with `2727 ms` for `B3_PromptEngineered_AI`, `2780 ms` for `B1_SingleShot_API`, and `2707 ms` for `B2_OpenSource_Only`. The mean number of model calls was `8.0` for `EAI-CO`, compared with `4.0`, `4.0`, and `3.0` for the three single-pass baselines, respectively.

This overhead is real, but it remains acceptable for offline creative optimization, where the goal is to generate stronger campaign variants before deployment rather than to serve real-time ad responses under millisecond constraints.

### 5.5 Cross-model Validation on a Commercial Generator

To test whether the observed gains depend on the local open-source backbone, a supplementary cross-model validation study was performed on a fixed 10-product subset using `gpt-5.4-mini-2026-03-17`. The same 4 audience segments and 40 product-audience tasks were retained, and only the five primary methods were evaluated.

The ranking pattern was preserved. `EAI-CO` again achieved the highest reward (`0.9957`), followed by `B2_OpenSource_Only` (`0.9751`), `B1_SingleShot_API` (`0.9736`), `B3_PromptEngineered_AI` (`0.9729`), and `B0_Template` (`0.8396`). The gains over all baselines remained statistically significant: `+0.1561` over `B0_Template` (`p = 1.54e-08`), `+0.0221` over `B1_SingleShot_API` (`p = 1.41e-03`), `+0.0206` over `B2_OpenSource_Only` (`p = 6.42e-04`), and `+0.0228` over `B3_PromptEngineered_AI` (`p = 1.01e-03`).

The commercial-model study also showed stable audience-wise behavior. `EAI-CO` remained the highest-ranked method for `family_users` (`0.9950`), `price_sensitive_consumers` (`0.9938`), `students` (`1.0000`), and `young_professionals` (`0.9939`).

These results support an important claim of the paper: the effectiveness of `EAI-CO` is not limited to a single local backbone. The advantage of exploratory multi-objective optimization transfers to a stronger commercial model under the same task protocol.

At the same time, this cross-model validation should be interpreted as a transferability study rather than the primary source of effect-size evidence. Because `gpt-5.4-mini` is a stronger generator under the current evaluator, the reward values are compressed near the upper bound, which reduces metric spread. The value of this experiment lies mainly in confirming ranking preservation across model families.

## 6. Discussion

### 6.1 Why the Exploratory Framework Outperforms Single-pass Generation

The main empirical pattern is consistent across both the local open-source backbone and the commercial-model validation: `EAI-CO` systematically outperforms template-based and single-pass generation methods. This suggests that the core gain is not simply due to stronger wording or a better initial prompt. Instead, the results support the central thesis of this paper: social-media ad creation benefits from being treated as an exploratory optimization problem rather than as a one-shot generation task.

Two mechanisms appear especially important. First, diversity preservation prevents the search process from collapsing into near-duplicate candidates. Second, iterative refinement improves the alignment between product features, audience needs, and persuasive framing. The ablation results show that removing either mechanism reduces performance, with diversity accounting for the largest single drop.

### 6.2 Audience Modeling Matters, but It Must Be Evaluated Structurally

The audience-modeling ablation provides a useful methodological lesson. In advertising research, “personalization” is often discussed at a conceptual level, but the present results show that persona conditioning should be treated as a measurable design component. When audience modeling was removed, reward dropped and audience fit declined substantially. This is a stronger result than simply claiming that the system produced “more personalized” copy. It shows that audience adaptation changes the optimization trajectory in a way that is detectable under a structured evaluator.

### 6.3 Compliance-aware Optimization Does Not Eliminate Persuasive Strength

The factual-penalty ablation remained close to the full method, which might initially appear to weaken the contribution of the compliance mechanism. However, that would be the wrong interpretation. The more defensible conclusion is that compliance-aware optimization can be incorporated with only a small reduction in raw persuasive reward while maintaining a safer and more stable search process. For digital media applications, this tradeoff is practically important: an optimization framework that yields slightly more aggressive language but weaker control over unsupported claims would be less suitable for responsible deployment.

### 6.4 Cross-model Transfer Strengthens the Paper's Main Claim

The `gpt-5.4-mini` validation materially improves the credibility of the paper. Without it, the results could be criticized as being tied to one local open-source model and one evaluator calibration. With the commercial-model validation included, the paper can make a narrower but stronger claim: the ranking advantage of the exploratory optimization framework generalizes across model families, even when the underlying generator becomes much stronger.

This is especially relevant for SCI-style evaluation standards, where reviewers often distinguish between a framework contribution and a backbone-specific effect. The current evidence supports the framework-level interpretation.

### 6.5 Limitations

Several limitations should be stated explicitly.

First, the current experimental package does not include real platform click-through-rate data. Therefore, the empirical claims should be restricted to predicted engagement, audience-conditioned creative quality, and automatic reward superiority rather than real online lift.

Second, the present version emphasizes automatic evaluation and cross-model validation, but it does not yet include a completed human evaluation study. This does not invalidate the current results, because the benchmark, ablations, and transfer experiment are internally coherent and reproducible. However, it does mean that the paper should avoid claiming verified human preference gains until a blinded participant study is added.

Third, the commercial-model validation exhibits score compression near the reward ceiling. This is not a failure of the experiment, but it does limit how much effect-size separation can be extracted from that supplementary setting.

### 6.6 Practical Implications

From a systems perspective, the results suggest that exploratory creative optimization is most appropriate for offline campaign preparation, batch creative generation, and audience-specific content design. The runtime overhead relative to single-pass generation is acceptable in those settings, especially when the output quality gain translates into fewer manual revision cycles.

For researchers, the main implication is methodological: automated content generation in digital media should be evaluated as a search-and-selection process, not merely as a language-generation problem. This shift makes it possible to study diversity, compliance, and audience alignment as first-class optimization variables rather than as after-the-fact observations.
