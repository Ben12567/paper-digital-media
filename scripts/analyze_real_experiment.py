from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "outputs" / "real_generation" / "real_candidates.csv"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "real_generation_analysis"))
    parser.add_argument("--model-label", default="local model")
    parser.add_argument("--protocol-label", default="5 primary methods, 4 ablations, offline automatic-evaluation design")
    return parser.parse_args()


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("method", as_index=False)
        .agg(
            n=("candidate_id", "count"),
            mean_reward=("reward", "mean"),
            mean_relevance=("relevance", "mean"),
            mean_clarity=("clarity", "mean"),
            mean_audience_fit=("audience_fit", "mean"),
            mean_diversity=("diversity", "mean"),
            mean_predicted_engagement=("predicted_engagement", "mean"),
            mean_brand_safety=("brand_safety", "mean"),
            mean_penalty=("factuality_penalty", "mean"),
            mean_model_calls=("model_calls", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
            median_latency_ms=("latency_ms", "median"),
        )
        .sort_values("mean_reward", ascending=False)
    )
    return summary.round(4)


def summarize_by_audience(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["audience_segment", "method"], as_index=False)
        .agg(
            n=("candidate_id", "count"),
            mean_reward=("reward", "mean"),
            mean_relevance=("relevance", "mean"),
            mean_clarity=("clarity", "mean"),
            mean_audience_fit=("audience_fit", "mean"),
            mean_predicted_engagement=("predicted_engagement", "mean"),
        )
        .sort_values(["audience_segment", "mean_reward"], ascending=[True, False])
    )
    return summary.round(4)


def bootstrap_mean_ci(values: pd.Series, seed: int, n_resamples: int = 10000) -> tuple[float, float]:
    data = values.to_numpy()
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(n_resamples, len(data)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running_max = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        corrected = min(1.0, (total - rank) * pvalues[index])
        running_max = max(running_max, corrected)
        adjusted[index] = running_max
    return adjusted.tolist()


def significance_vs_ours(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ours = (
        df[df["method"] == "Ours_EAI_CO"][["task_id", "reward", "predicted_engagement", "audience_fit"]]
        .rename(
            columns={
                "reward": "ours_reward",
                "predicted_engagement": "ours_predicted_engagement",
                "audience_fit": "ours_audience_fit",
            }
        )
    )
    for method_index, method in enumerate(PRIMARY_METHODS):
        if method == "Ours_EAI_CO":
            continue
        baseline = (
            df[df["method"] == method][["task_id", "reward", "predicted_engagement", "audience_fit"]]
            .rename(
                columns={
                    "reward": "baseline_reward",
                    "predicted_engagement": "baseline_predicted_engagement",
                    "audience_fit": "baseline_audience_fit",
                }
            )
        )
        merged = ours.merge(baseline, on="task_id", how="inner")
        reward_diff = merged["ours_reward"] - merged["baseline_reward"]
        reward_test = stats.ttest_rel(merged["ours_reward"], merged["baseline_reward"])
        engage_test = stats.ttest_rel(
            merged["ours_predicted_engagement"],
            merged["baseline_predicted_engagement"],
        )
        fit_test = stats.ttest_rel(
            merged["ours_audience_fit"],
            merged["baseline_audience_fit"],
        )
        ci_lower, ci_upper = bootstrap_mean_ci(reward_diff, seed=42 + method_index)
        reward_std = reward_diff.std(ddof=1)
        rows.append(
            {
                "baseline": method,
                "n_tasks": len(merged),
                "reward_delta": round(reward_diff.mean(), 4),
                "reward_ci_lower": round(ci_lower, 4),
                "reward_ci_upper": round(ci_upper, 4),
                "reward_pvalue": reward_test.pvalue,
                "reward_cohens_d": round(reward_diff.mean() / reward_std, 4),
                "predicted_engagement_delta": round(
                    (merged["ours_predicted_engagement"] - merged["baseline_predicted_engagement"]).mean(), 4
                ),
                "predicted_engagement_pvalue": engage_test.pvalue,
                "audience_fit_delta": round((merged["ours_audience_fit"] - merged["baseline_audience_fit"]).mean(), 4),
                "audience_fit_pvalue": fit_test.pvalue,
            }
        )
    result = pd.DataFrame(rows)
    result["reward_adjusted_pvalue"] = holm_adjust(result["reward_pvalue"].tolist())
    return result


def ablation_summary(df: pd.DataFrame) -> pd.DataFrame:
    methods = ["Ours_EAI_CO"] + ABLATION_METHODS
    subset = df[df["method"].isin(methods)].copy()
    summary = summarize(subset)
    ours_row = summary[summary["method"] == "Ours_EAI_CO"].iloc[0]
    summary["reward_gap_vs_ours"] = (summary["mean_reward"] - ours_row["mean_reward"]).round(4)
    summary["audience_fit_gap_vs_ours"] = (summary["mean_audience_fit"] - ours_row["mean_audience_fit"]).round(4)
    return summary


def cost_summary(df: pd.DataFrame) -> pd.DataFrame:
    costs = (
        df.groupby("method", as_index=False)
        .agg(
            mean_model_calls=("model_calls", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
            median_latency_ms=("latency_ms", "median"),
            total_latency_hours=("latency_ms", lambda s: s.sum() / 1000 / 3600),
        )
        .sort_values("mean_latency_ms", ascending=False)
    )
    return costs.round(4)


def representative_cases(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = df[df["method"].isin(PRIMARY_METHODS)].copy()
    top = primary.sort_values("reward", ascending=False).head(10)
    weak = primary.sort_values("reward", ascending=True).head(10)
    cols = [
        "task_id",
        "method",
        "product_id",
        "audience_segment",
        "reward",
        "relevance",
        "clarity",
        "audience_fit",
        "predicted_engagement",
        "headline",
        "body",
        "cta",
        "visual_prompt",
    ]
    return top[cols], weak[cols]


def write_report(
    out_dir: Path,
    summary: pd.DataFrame,
    audience: pd.DataFrame,
    sig: pd.DataFrame,
    ablation: pd.DataFrame,
    costs: pd.DataFrame,
    model_label: str,
    protocol_label: str,
) -> None:
    primary = summary[summary["method"].isin(PRIMARY_METHODS)].copy()
    best = primary.sort_values("mean_reward", ascending=False).iloc[0]
    audience_best = audience[audience["method"] == "Ours_EAI_CO"]
    text = [
        "# Local GPU Experiment Report",
        "",
        "## Setup",
        f"- Local generator: `{model_label}`",
        "- Evaluation: automatic multi-objective metrics with full benchmark coverage",
        f"- Protocol: {protocol_label}",
        "",
        "## Main Findings",
        f"- Best primary method: `{best['method']}` with mean reward `{best['mean_reward']:.4f}`.",
        f"- `Ours_EAI_CO` mean predicted engagement: `{primary.loc[primary['method'] == 'Ours_EAI_CO', 'mean_predicted_engagement'].iloc[0]:.4f}`.",
        f"- `Ours_EAI_CO` mean latency: `{primary.loc[primary['method'] == 'Ours_EAI_CO', 'mean_latency_ms'].iloc[0]:.2f}` ms per task.",
        f"- `Ours_EAI_CO` median latency: `{primary.loc[primary['method'] == 'Ours_EAI_CO', 'median_latency_ms'].iloc[0]:.2f}` ms per task.",
        "",
        "## Audience Breakdown",
    ]
    for _, row in audience_best.iterrows():
        text.append(
            f"- {row['audience_segment']}: reward `{row['mean_reward']:.4f}`, "
            f"audience fit `{row['mean_audience_fit']:.4f}`, predicted engagement `{row['mean_predicted_engagement']:.4f}`."
        )
    text.extend(
        [
            "",
            "## Statistical Comparison vs Ours",
        ]
    )
    for _, row in sig.iterrows():
        text.append(
            f"- vs `{row['baseline']}`: reward delta `{row['reward_delta']:.4f}` "
            f"(95% CI [{row['reward_ci_lower']:.4f}, {row['reward_ci_upper']:.4f}], "
            f"p={row['reward_pvalue']:.4g}, adjusted p={row['reward_adjusted_pvalue']:.4g}, "
            f"Cohen's d={row['reward_cohens_d']:.4f}), "
            f"engagement delta `{row['predicted_engagement_delta']:.4f}` (p={row['predicted_engagement_pvalue']:.4g}), "
            f"audience-fit delta `{row['audience_fit_delta']:.4f}` (p={row['audience_fit_pvalue']:.4g})."
        )
    text.extend(
        [
            "",
            "## Ablations",
        ]
    )
    for _, row in ablation.iterrows():
        text.append(
            f"- `{row['method']}`: reward `{row['mean_reward']:.4f}`, "
            f"reward gap vs ours `{row['reward_gap_vs_ours']:.4f}`, "
            f"audience-fit gap `{row['audience_fit_gap_vs_ours']:.4f}`."
        )
    text.extend(
        [
            "",
            "## Cost Profile",
        ]
    )
    for _, row in costs.iterrows():
        text.append(
            f"- `{row['method']}`: mean latency `{row['mean_latency_ms']:.2f}` ms, "
            f"median latency `{row['median_latency_ms']:.2f}` ms, "
            f"mean model calls `{row['mean_model_calls']:.2f}`, "
            f"aggregate latency `{row['total_latency_hours']:.4f}` hours."
        )
    (out_dir / "local_gpu_experiment_report.md").write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    summary = summarize(df)
    primary_summary = summarize(df[df["method"].isin(PRIMARY_METHODS)])
    audience = summarize_by_audience(df[df["method"].isin(PRIMARY_METHODS)])
    sig = significance_vs_ours(df[df["method"].isin(PRIMARY_METHODS)])
    ablation = ablation_summary(df)
    costs = cost_summary(df)
    top_cases, weak_cases = representative_cases(df)

    summary.to_csv(out_dir / "method_summary.csv", index=False, encoding="utf-8-sig")
    primary_summary.to_csv(out_dir / "primary_method_summary.csv", index=False, encoding="utf-8-sig")
    audience.to_csv(out_dir / "audience_summary.csv", index=False, encoding="utf-8-sig")
    sig.to_csv(out_dir / "automatic_significance_vs_ours.csv", index=False, encoding="utf-8-sig")
    ablation.to_csv(out_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")
    costs.to_csv(out_dir / "cost_summary.csv", index=False, encoding="utf-8-sig")
    top_cases.to_csv(out_dir / "top_cases.csv", index=False, encoding="utf-8-sig")
    weak_cases.to_csv(out_dir / "weak_cases.csv", index=False, encoding="utf-8-sig")
    write_report(out_dir, summary, audience, sig, ablation, costs, args.model_label, args.protocol_label)
    print(f"Analysis written to {out_dir}")


if __name__ == "__main__":
    main()
