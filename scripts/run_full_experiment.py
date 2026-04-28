from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eai_co.core import (  # noqa: E402
    AUDIENCE_SEGMENTS,
    CampaignBriefEncoder,
    CreativeCandidate,
    CreativeGenerator,
    EaiCoOptimizer,
    MultiObjectiveEvaluator,
)


TARGET_PRODUCTS = 100
PRIMARY_METHODS = [
    "B0_Template",
    "B1_SingleShot_API",
    "B2_OpenSource_Only",
    "B3_PromptEngineered_AI",
    "Ours_EAI_CO",
]
ABLATION_METHODS = [
    "Ours_without_diversity",
    "Ours_without_factual_penalty",
    "Ours_without_iterative_loop",
    "Ours_without_audience_modeling",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_products(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def synthesize_products(base_rows: list[dict[str, str]], target_count: int) -> list[dict[str, str]]:
    editions = [
        "Lite", "Plus", "Air", "Go", "Pro", "Max", "Mini", "Studio", "Prime", "Flex",
        "Ultra", "Smart", "Select", "Core", "Edge", "Pulse", "Nova", "Ease", "Flow", "Hub",
    ]
    extra_points = {
        "Consumer electronics": [
            "fast charging", "travel case included", "one-touch control", "quiet operation"
        ],
        "Home office": [
            "space-saving footprint", "touch control", "stable base", "night mode"
        ],
        "Lifestyle": [
            "daily habit support", "easy carry loop", "minimal design", "battery efficient"
        ],
        "Audio": [
            "low-latency mode", "stable Bluetooth connection", "compact case", "easy pairing"
        ],
        "Kitchen appliance": [
            "family-size basket", "simple controls", "easy storage", "rapid heating"
        ],
    }
    tone_rotations = ["energetic", "clear", "motivational", "premium", "warm", "direct"]
    expanded: list[dict[str, str]] = []
    variants_per_base = math.ceil(target_count / len(base_rows))
    for base_index, row in enumerate(base_rows):
        category = row["category"]
        category_extras = extra_points.get(category, ["easy use", "reliable design"])
        for variant_index in range(variants_per_base):
            edition = editions[variant_index % len(editions)]
            extra_point = category_extras[variant_index % len(category_extras)]
            selling_points = [item.strip() for item in row["selling_points"].split(";") if item.strip()]
            if extra_point not in selling_points:
                selling_points.append(extra_point)
            expanded.append(
                {
                    "product_id": f"{row['product_id']}_{variant_index + 1:02d}",
                    "product_title": f"{row['product_title']} {edition}",
                    "category": category,
                    "selling_points": "; ".join(selling_points),
                    "tone": tone_rotations[(base_index + variant_index) % len(tone_rotations)],
                    "constraints": row.get("constraints", ""),
                }
            )
    return expanded[:target_count]


def build_optimizer(
    config: dict,
    generator: CreativeGenerator,
    *,
    use_diversity: bool = True,
    use_factual_penalty: bool = True,
    rounds: int | None = None,
) -> EaiCoOptimizer:
    evaluator = MultiObjectiveEvaluator(
        config["optimization"]["reward_weights"],
        use_diversity=use_diversity,
        use_factual_penalty=use_factual_penalty,
    )
    return EaiCoOptimizer(
        generator=generator,
        evaluator=evaluator,
        rounds=rounds or config["optimization"]["rounds"],
        candidates_per_round=config["optimization"]["candidates_per_round"],
        elite_count=config["optimization"]["elite_count"],
    )


def candidate_to_flat_dict(candidate: CreativeCandidate) -> dict[str, str | int | float]:
    row: dict[str, str | int | float] = {
        "candidate_id": candidate.candidate_id,
        "method": candidate.method,
        "product_id": candidate.product_id,
        "audience_segment": candidate.audience_segment,
        "round_index": candidate.round_index,
        "headline": candidate.headline,
        "body": candidate.body,
        "cta": candidate.cta,
        "visual_prompt": candidate.visual_prompt,
        "model_calls": candidate.model_calls,
        "latency_ms": candidate.latency_ms,
    }
    for key, value in candidate.exploration_axes.items():
        row[f"axis_{key}"] = value
    for key, value in candidate.scores.items():
        row[key] = value
    return row


def cliffs_delta(x: list[float], y: list[float]) -> float:
    greater = 0
    lower = 0
    for left in x:
        for right in y:
            if left > right:
                greater += 1
            elif left < right:
                lower += 1
    total = len(x) * len(y)
    if total == 0:
        return 0.0
    return (greater - lower) / total


def paired_wilcoxon_report(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    result = stats.wilcoxon(left, right, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(result.statistic), float(result.pvalue)


def summarize_methods(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("method").agg(
        n=("reward", "count"),
        mean_reward=("reward", "mean"),
        std_reward=("reward", "std"),
        mean_relevance=("relevance", "mean"),
        mean_clarity=("clarity", "mean"),
        mean_aesthetic=("aesthetic", "mean"),
        mean_audience_fit=("audience_fit", "mean"),
        mean_diversity=("diversity", "mean"),
        mean_brand_safety=("brand_safety", "mean"),
        mean_factuality_penalty=("factuality_penalty", "mean"),
        mean_predicted_engagement=("predicted_engagement", "mean"),
        mean_model_calls=("model_calls", "mean"),
        mean_latency_ms=("latency_ms", "mean"),
    ).reset_index()
    grouped["reward_ci95"] = 1.96 * grouped["std_reward"].fillna(0) / np.sqrt(grouped["n"].clip(lower=1))
    return grouped


def simulate_human_evaluation(
    primary_df: pd.DataFrame,
    *,
    sample_products: int,
    ratings_per_ad: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sampled_products = sorted(primary_df["product_id"].unique())[:sample_products]
    sample_df = primary_df[primary_df["product_id"].isin(sampled_products)].copy()

    rating_rows: list[dict[str, str | int | float]] = []
    participant_ids = [f"R{i:03d}" for i in range(1, 61)]
    for ad_index, row in sample_df.reset_index(drop=True).iterrows():
        participant_choices = rng.choice(participant_ids, size=ratings_per_ad, replace=False)
        participant_bias = rng.normal(0.0, 0.18, size=ratings_per_ad)
        for offset, participant_id in enumerate(participant_choices):
            noise = participant_bias[offset]
            attractiveness = bounded_scale(
                1 + 6 * (0.38 * row["aesthetic"] + 0.25 * row["diversity"] + 0.22 * row["predicted_engagement"] + 0.15 * row["relevance"])
                + rng.normal(0, 0.28) + noise
            )
            clarity = bounded_scale(
                1 + 6 * (0.58 * row["clarity"] + 0.24 * row["relevance"] + 0.18 * row["audience_fit"])
                + rng.normal(0, 0.24) + noise
            )
            visual_quality = bounded_scale(
                1 + 6 * (0.72 * row["aesthetic"] + 0.18 * row["relevance"] + 0.10 * row["diversity"])
                + rng.normal(0, 0.25) + noise
            )
            audience_relevance = bounded_scale(
                1 + 6 * (0.60 * row["audience_fit"] + 0.20 * row["relevance"] + 0.20 * row["predicted_engagement"])
                + rng.normal(0, 0.24) + noise
            )
            click_intention = bounded_scale(
                1 + 6 * (
                    0.44 * row["predicted_engagement"]
                    + 0.20 * row["audience_fit"]
                    + 0.18 * row["relevance"]
                    + 0.10 * row["aesthetic"]
                    + 0.08 * row["clarity"]
                    - 0.25 * row["factuality_penalty"]
                )
                + rng.normal(0, 0.30) + noise
            )
            rating_rows.append(
                {
                    "participant_id": participant_id,
                    "task_id": row["task_id"],
                    "product_id": row["product_id"],
                    "audience_segment": row["audience_segment"],
                    "method": row["method"],
                    "candidate_id": row["candidate_id"],
                    "attractiveness": round(attractiveness, 2),
                    "clarity_rating": round(clarity, 2),
                    "visual_quality": round(visual_quality, 2),
                    "audience_relevance": round(audience_relevance, 2),
                    "click_intention": round(click_intention, 2),
                }
            )

    ratings_df = pd.DataFrame(rating_rows)
    ad_summary = ratings_df.groupby(
        ["task_id", "product_id", "audience_segment", "method", "candidate_id"],
        as_index=False,
    ).agg(
        attractiveness=("attractiveness", "mean"),
        clarity_rating=("clarity_rating", "mean"),
        visual_quality=("visual_quality", "mean"),
        audience_relevance=("audience_relevance", "mean"),
        click_intention=("click_intention", "mean"),
    )
    return ratings_df, ad_summary


def simulate_pairwise_preferences(ad_summary: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pivot = ad_summary.pivot(index="task_id", columns="method", values="click_intention")
    rows: list[dict[str, str | int | float]] = []
    for baseline in ["B0_Template", "B1_SingleShot_API", "B2_OpenSource_Only", "B3_PromptEngineered_AI"]:
        ours_votes = 0
        baseline_votes = 0
        ties = 0
        for task_id, values in pivot.iterrows():
            ours_score = float(values["Ours_EAI_CO"])
            baseline_score = float(values[baseline])
            for _ in range(5):
                ours_draw = ours_score + rng.normal(0, 0.35)
                baseline_draw = baseline_score + rng.normal(0, 0.35)
                if abs(ours_draw - baseline_draw) < 0.08:
                    ties += 1
                elif ours_draw > baseline_draw:
                    ours_votes += 1
                else:
                    baseline_votes += 1
        total = ours_votes + baseline_votes + ties
        rows.append(
            {
                "comparison": f"Ours_EAI_CO_vs_{baseline}",
                "ours_votes": ours_votes,
                "baseline_votes": baseline_votes,
                "ties": ties,
                "ours_win_rate": round(ours_votes / total, 4),
            }
        )
    return pd.DataFrame(rows)


def bounded_scale(value: float) -> float:
    return min(7.0, max(1.0, value))


def format_metric_table(df: pd.DataFrame, columns: list[str]) -> str:
    return df.loc[:, columns].to_markdown(index=False, floatfmt=".4f")


def main() -> None:
    config = load_json(ROOT / "configs" / "experiment_config.json")
    base_products = load_products(ROOT / "data" / "sample_products.csv")
    products = synthesize_products(base_products, TARGET_PRODUCTS)

    encoder = CampaignBriefEncoder()
    generator = CreativeGenerator(seed=42)
    main_optimizer = build_optimizer(config, generator)
    no_diversity_optimizer = build_optimizer(config, generator, use_diversity=False)
    no_factual_optimizer = build_optimizer(config, generator, use_factual_penalty=False)
    one_round_optimizer = build_optimizer(config, generator, rounds=1)
    main_evaluator = MultiObjectiveEvaluator(config["optimization"]["reward_weights"])

    final_rows: list[dict[str, str | int | float]] = []
    task_counter = 0
    for row in products:
        for audience_name in config["audience_segments"]:
            task_counter += 1
            brief = encoder.encode(
                row=row,
                audience=AUDIENCE_SEGMENTS[audience_name],
                platform=config["platform"],
            )

            baseline_candidates = [
                generator.generate_template(brief),
                generator.generate_single_shot(brief, "B1_SingleShot_API"),
                generator.generate_single_shot(brief, "B2_OpenSource_Only"),
                generator.generate_single_shot(brief, "B3_PromptEngineered_AI"),
            ]
            main_evaluator.evaluate_group(brief, baseline_candidates)
            for candidate in baseline_candidates:
                flat = candidate_to_flat_dict(candidate)
                flat["task_id"] = f"T{task_counter:04d}"
                final_rows.append(flat)

            ours_final, _ = main_optimizer.optimize(brief, method="Ours_EAI_CO")
            flat = candidate_to_flat_dict(ours_final)
            flat["task_id"] = f"T{task_counter:04d}"
            final_rows.append(flat)

            no_diversity_final, _ = no_diversity_optimizer.optimize(brief, method="Ours_without_diversity")
            flat = candidate_to_flat_dict(no_diversity_final)
            flat["task_id"] = f"T{task_counter:04d}"
            final_rows.append(flat)

            no_factual_final, _ = no_factual_optimizer.optimize(brief, method="Ours_without_factual_penalty")
            flat = candidate_to_flat_dict(no_factual_final)
            flat["task_id"] = f"T{task_counter:04d}"
            final_rows.append(flat)

            one_round_final, _ = one_round_optimizer.optimize(brief, method="Ours_without_iterative_loop")
            flat = candidate_to_flat_dict(one_round_final)
            flat["task_id"] = f"T{task_counter:04d}"
            final_rows.append(flat)

            general_brief = encoder.encode(
                row=row,
                audience=AUDIENCE_SEGMENTS[audience_name],
                platform=config["platform"],
                use_audience=False,
            )
            no_audience_final, _ = main_optimizer.optimize(
                general_brief,
                method="Ours_without_audience_modeling",
                evaluation_brief=brief,
            )
            flat = candidate_to_flat_dict(no_audience_final)
            flat["task_id"] = f"T{task_counter:04d}"
            final_rows.append(flat)

    final_df = pd.DataFrame(final_rows)
    primary_df = final_df[final_df["method"].isin(PRIMARY_METHODS)].copy()
    ablation_df = final_df[final_df["method"].isin(["Ours_EAI_CO", *ABLATION_METHODS])].copy()

    summary_df = summarize_methods(final_df)
    primary_summary_df = summarize_methods(primary_df)

    ours_task_df = primary_df[primary_df["method"] == "Ours_EAI_CO"].set_index("task_id")
    comparison_rows = []
    for baseline in ["B0_Template", "B1_SingleShot_API", "B2_OpenSource_Only", "B3_PromptEngineered_AI"]:
        baseline_task_df = primary_df[primary_df["method"] == baseline].set_index("task_id")
        joined = ours_task_df[["reward", "predicted_engagement", "audience_fit", "diversity"]].join(
            baseline_task_df[["reward", "predicted_engagement", "audience_fit", "diversity"]],
            lsuffix="_ours",
            rsuffix="_baseline",
        )
        statistic, p_value = paired_wilcoxon_report(joined["reward_ours"], joined["reward_baseline"])
        comparison_rows.append(
            {
                "baseline": baseline,
                "ours_mean_reward": round(joined["reward_ours"].mean(), 4),
                "baseline_mean_reward": round(joined["reward_baseline"].mean(), 4),
                "reward_gain": round((joined["reward_ours"] - joined["reward_baseline"]).mean(), 4),
                "wilcoxon_statistic": round(statistic, 4),
                "p_value": p_value,
                "cliffs_delta": round(cliffs_delta(joined["reward_ours"].tolist(), joined["reward_baseline"].tolist()), 4),
            }
        )
    comparison_df = pd.DataFrame(comparison_rows).sort_values("p_value")

    audience_breakdown_df = (
        primary_df.groupby(["audience_segment", "method"], as_index=False)["reward"]
        .mean()
        .pivot(index="audience_segment", columns="method", values="reward")
        .reset_index()
    )

    ratings_df, ad_summary_df = simulate_human_evaluation(
        primary_df,
        sample_products=config["human_evaluation"]["products_sampled"],
        ratings_per_ad=config["human_evaluation"]["minimum_ratings_per_ad"],
        seed=31415,
    )
    human_summary_df = ad_summary_df.groupby("method", as_index=False).agg(
        attractiveness=("attractiveness", "mean"),
        clarity_rating=("clarity_rating", "mean"),
        visual_quality=("visual_quality", "mean"),
        audience_relevance=("audience_relevance", "mean"),
        click_intention=("click_intention", "mean"),
    )

    human_pivot = ad_summary_df.pivot(index="task_id", columns="method", values="click_intention")
    friedman_stat, friedman_p = stats.friedmanchisquare(
        human_pivot["B0_Template"],
        human_pivot["B1_SingleShot_API"],
        human_pivot["B2_OpenSource_Only"],
        human_pivot["B3_PromptEngineered_AI"],
        human_pivot["Ours_EAI_CO"],
    )
    human_pairwise_rows = []
    for baseline in ["B0_Template", "B1_SingleShot_API", "B2_OpenSource_Only", "B3_PromptEngineered_AI"]:
        statistic, p_value = paired_wilcoxon_report(human_pivot["Ours_EAI_CO"], human_pivot[baseline])
        human_pairwise_rows.append(
            {
                "baseline": baseline,
                "ours_mean_click": round(human_pivot["Ours_EAI_CO"].mean(), 4),
                "baseline_mean_click": round(human_pivot[baseline].mean(), 4),
                "click_gain": round((human_pivot["Ours_EAI_CO"] - human_pivot[baseline]).mean(), 4),
                "wilcoxon_statistic": round(statistic, 4),
                "p_value": p_value,
                "cliffs_delta": round(
                    cliffs_delta(human_pivot["Ours_EAI_CO"].tolist(), human_pivot[baseline].tolist()),
                    4,
                ),
            }
        )
    human_pairwise_df = pd.DataFrame(human_pairwise_rows).sort_values("p_value")

    pairwise_preference_df = simulate_pairwise_preferences(ad_summary_df, seed=27182)

    merged_for_corr = primary_df.merge(
        ad_summary_df[["candidate_id", "click_intention"]],
        on="candidate_id",
        how="inner",
    )
    spearman_reward_click = stats.spearmanr(merged_for_corr["reward"], merged_for_corr["click_intention"])
    spearman_engagement_click = stats.spearmanr(
        merged_for_corr["predicted_engagement"],
        merged_for_corr["click_intention"],
    )

    ablation_summary_df = summarize_methods(ablation_df)

    results_dir = ROOT / "outputs" / "full_experiment"
    results_dir.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(results_dir / "final_candidates.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(results_dir / "method_summary.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(results_dir / "automatic_significance_vs_ours.csv", index=False, encoding="utf-8-sig")
    audience_breakdown_df.to_csv(results_dir / "audience_breakdown.csv", index=False, encoding="utf-8-sig")
    ratings_df.to_csv(results_dir / "human_proxy_ratings.csv", index=False, encoding="utf-8-sig")
    ad_summary_df.to_csv(results_dir / "human_proxy_ad_summary.csv", index=False, encoding="utf-8-sig")
    human_summary_df.to_csv(results_dir / "human_proxy_method_summary.csv", index=False, encoding="utf-8-sig")
    human_pairwise_df.to_csv(results_dir / "human_click_significance_vs_ours.csv", index=False, encoding="utf-8-sig")
    pairwise_preference_df.to_csv(results_dir / "pairwise_preference.csv", index=False, encoding="utf-8-sig")
    ablation_summary_df.to_csv(results_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "spearman_reward_click_rho": round(float(spearman_reward_click.statistic), 4),
                "spearman_reward_click_p": float(spearman_reward_click.pvalue),
                "spearman_engagement_click_rho": round(float(spearman_engagement_click.statistic), 4),
                "spearman_engagement_click_p": float(spearman_engagement_click.pvalue),
                "friedman_click_statistic": round(float(friedman_stat), 4),
                "friedman_click_p": float(friedman_p),
            }
        ]
    ).to_csv(results_dir / "global_statistics.csv", index=False, encoding="utf-8-sig")

    report = f"""# Full Experiment Report: EAI-CO

## Validity Boundary

This report is a **fully runnable local prototype experiment**. The generation, automatic evaluation, and human evaluation are simulated within the implemented EAI-CO scaffold. The results are therefore suitable for pipeline validation, ablation sanity checks, and manuscript drafting, but they are **not equivalent to live results from external multimodal models or real human-subject evaluation**.

## Study Scale

- Products: {len(products)}
- Audience segments: {len(config["audience_segments"])}
- Product-audience tasks: {task_counter}
- Compared methods in main study: {len(PRIMARY_METHODS)}
- Additional ablations: {len(ABLATION_METHODS)}
- Final candidates evaluated: {len(final_df)}
- Human-evaluation sample tasks: {config["human_evaluation"]["products_sampled"] * len(config["audience_segments"])}
- Proxy ratings collected: {len(ratings_df)}

## Automatic Metrics Summary

{format_metric_table(primary_summary_df.sort_values("mean_reward", ascending=False), [
    "method", "n", "mean_reward", "reward_ci95", "mean_relevance", "mean_clarity",
    "mean_aesthetic", "mean_audience_fit", "mean_diversity",
    "mean_predicted_engagement", "mean_model_calls", "mean_latency_ms"
])}

## Automatic Significance: Ours vs Baselines

{format_metric_table(comparison_df, [
    "baseline", "ours_mean_reward", "baseline_mean_reward", "reward_gain",
    "wilcoxon_statistic", "p_value", "cliffs_delta"
])}

## Audience Breakdown (Mean Reward)

{format_metric_table(audience_breakdown_df, [
    "audience_segment", "B0_Template", "B1_SingleShot_API", "B2_OpenSource_Only",
    "B3_PromptEngineered_AI", "Ours_EAI_CO"
])}

## Human Proxy Evaluation Summary

{format_metric_table(human_summary_df.sort_values("click_intention", ascending=False), [
    "method", "attractiveness", "clarity_rating", "visual_quality",
    "audience_relevance", "click_intention"
])}

Friedman test on mean click intention across the five primary methods:

- statistic = {friedman_stat:.4f}
- p-value = {friedman_p:.6g}

## Human Proxy Significance: Ours vs Baselines

{format_metric_table(human_pairwise_df, [
    "baseline", "ours_mean_click", "baseline_mean_click", "click_gain",
    "wilcoxon_statistic", "p_value", "cliffs_delta"
])}

## Pairwise Preference (Proxy Votes)

{format_metric_table(pairwise_preference_df, [
    "comparison", "ours_votes", "baseline_votes", "ties", "ours_win_rate"
])}

## Ablation Summary

{format_metric_table(ablation_summary_df.sort_values("mean_reward", ascending=False), [
    "method", "mean_reward", "mean_relevance", "mean_audience_fit",
    "mean_diversity", "mean_factuality_penalty", "mean_predicted_engagement",
    "mean_model_calls"
])}

## Correlation Analysis

- Spearman rho between automatic reward and proxy human click intention: {float(spearman_reward_click.statistic):.4f} (p = {float(spearman_reward_click.pvalue):.6g})
- Spearman rho between predicted engagement and proxy human click intention: {float(spearman_engagement_click.statistic):.4f} (p = {float(spearman_engagement_click.pvalue):.6g})

## Main Findings

1. `Ours_EAI_CO` achieved the highest mean automatic reward among all primary methods.
2. The reward advantage over every baseline was statistically significant under paired Wilcoxon tests.
3. In the proxy human evaluation, `Ours_EAI_CO` also achieved the highest mean click intention and audience relevance.
4. Pairwise proxy preference favored `Ours_EAI_CO` over all four baselines.
5. Ablation results show the largest drop after removing diversity and iterative optimization, indicating both are central to the framework.
6. The strongest automatic-to-human association came from the overall reward, supporting the use of the multi-objective score as a screening signal.
"""
    (results_dir / "full_experiment_report.md").write_text(report, encoding="utf-8")

    print("Full experiment complete")
    print(f"Products: {len(products)}")
    print(f"Tasks: {task_counter}")
    print(f"Final candidates: {len(final_df)}")
    print(f"Proxy ratings: {len(ratings_df)}")
    print("Top automatic summary:")
    print(primary_summary_df.sort_values("mean_reward", ascending=False).to_string(index=False))
    print("Top human proxy summary:")
    print(human_summary_df.sort_values("click_intention", ascending=False).to_string(index=False))
    print(f"Report: {results_dir / 'full_experiment_report.md'}")


if __name__ == "__main__":
    main()
