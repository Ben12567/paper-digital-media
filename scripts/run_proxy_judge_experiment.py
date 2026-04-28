from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eai_co.judge import (  # noqa: E402
    HuggingFaceJudgeProvider,
    TransformersJudgeProvider,
    build_checklist_prompt,
    build_pairwise_prompt,
)


PRIMARY_METHODS = [
    "B0_Template",
    "B1_SingleShot_API",
    "B2_OpenSource_Only",
    "B3_PromptEngineered_AI",
    "Ours_EAI_CO",
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def majority_vote(a: str, b: str) -> str:
    if a == b:
        return a
    return "Tie"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--judge-config", default=str(ROOT / "configs" / "proxy_judge_config.json"))
    parser.add_argument("--limit-tasks", type=int, default=None)
    args = parser.parse_args()

    cfg = load_json(Path(args.judge_config))
    df = pd.read_csv(args.input)
    df = df[df["method"].isin(PRIMARY_METHODS)].copy()
    if args.limit_tasks:
        keep_tasks = sorted(df["task_id"].unique())[: args.limit_tasks]
        df = df[df["task_id"].isin(keep_tasks)].copy()

    judge_cfg = cfg["judge_model"]
    if judge_cfg["kind"] == "huggingface_inference":
        judge = HuggingFaceJudgeProvider(
            model=judge_cfg["model"],
            provider=judge_cfg.get("provider", "hf-inference"),
            temperature=judge_cfg.get("temperature", 0.0),
            max_tokens=judge_cfg.get("max_tokens", 500),
        )
    elif judge_cfg["kind"] == "transformers":
        judge = TransformersJudgeProvider(
            model_name_or_path=judge_cfg["model_name_or_path"],
            temperature=judge_cfg.get("temperature", 0.0),
            max_new_tokens=judge_cfg.get("max_tokens", 300),
        )
    else:
        raise ValueError(f"Unsupported judge kind: {judge_cfg['kind']}")

    checklist_rows = []
    for _, row in df.iterrows():
        prompt_default = build_checklist_prompt(row.to_dict(), prompt_variant="default")
        prompt_strict = build_checklist_prompt(row.to_dict(), prompt_variant="strict")
        default_result = judge.score_checklist(prompt_default)
        strict_result = judge.score_checklist(prompt_strict)
        default_dict = default_result.to_dict()
        strict_dict = strict_result.to_dict()
        checklist_rows.append(
            {
                "task_id": row["task_id"],
                "candidate_id": row["candidate_id"],
                "method": row["method"],
                "product_id": row["product_id"],
                "audience_segment": row["audience_segment"],
                "automatic_reward": row.get("reward"),
                "judge_default_overall": default_dict["overall_score"],
                "judge_strict_overall": strict_dict["overall_score"],
                "judge_default_checklist_sum": default_dict["checklist_sum"],
                "judge_strict_checklist_sum": strict_dict["checklist_sum"],
                "judge_overall_mean": round((default_dict["overall_score"] + strict_dict["overall_score"]) / 2, 4),
                "judge_checklist_mean": round((default_dict["checklist_sum"] + strict_dict["checklist_sum"]) / 2, 4),
                "judge_prompt_agreement": int(default_dict["overall_score"] == strict_dict["overall_score"]),
                "default_reason": default_dict["reason"],
                "strict_reason": strict_dict["reason"],
            }
        )

    checklist_df = pd.DataFrame(checklist_rows)
    checklist_summary = checklist_df.groupby("method", as_index=False).agg(
        n=("candidate_id", "count"),
        judge_overall_mean=("judge_overall_mean", "mean"),
        judge_checklist_mean=("judge_checklist_mean", "mean"),
        automatic_reward_mean=("automatic_reward", "mean"),
        judge_prompt_agreement=("judge_prompt_agreement", "mean"),
    )

    ours_df = df[df["method"] == "Ours_EAI_CO"].set_index("task_id")
    pairwise_rows = []
    baselines = ["B0_Template", "B1_SingleShot_API", "B2_OpenSource_Only", "B3_PromptEngineered_AI"]
    for baseline in baselines:
        baseline_df = df[df["method"] == baseline].set_index("task_id")
        common_tasks = sorted(set(ours_df.index) & set(baseline_df.index))
        for task_id in common_tasks:
            ours_row = ours_df.loc[task_id].to_dict()
            baseline_row = baseline_df.loc[task_id].to_dict()
            forward = judge.score_pairwise(build_pairwise_prompt(ours_row, baseline_row))
            reverse = judge.score_pairwise(build_pairwise_prompt(baseline_row, ours_row))
            reverse_mapped_winner = {"A": "B", "B": "A", "Tie": "Tie"}[reverse.winner]
            final_winner = majority_vote(forward.winner, reverse_mapped_winner)
            pairwise_rows.append(
                {
                    "task_id": task_id,
                    "baseline": baseline,
                    "forward_winner": forward.winner,
                    "reverse_winner_mapped": reverse_mapped_winner,
                    "final_winner": final_winner,
                    "forward_confidence": forward.confidence,
                    "reverse_confidence": reverse.confidence,
                    "order_consistent": int(forward.winner == reverse_mapped_winner),
                }
            )

    pairwise_df = pd.DataFrame(pairwise_rows)
    pairwise_summary_rows = []
    if not pairwise_df.empty:
        for baseline, group in pairwise_df.groupby("baseline"):
            ours_wins = int((group["final_winner"] == "A").sum())
            baseline_wins = int((group["final_winner"] == "B").sum())
            ties = int((group["final_winner"] == "Tie").sum())
            total = len(group)
            pairwise_summary_rows.append(
                {
                    "baseline": baseline,
                    "ours_wins": ours_wins,
                    "baseline_wins": baseline_wins,
                    "ties": ties,
                    "ours_win_rate": round(ours_wins / total, 4) if total else 0.0,
                    "order_consistency": round(group["order_consistent"].mean(), 4),
                    "mean_confidence": round(
                        statistics.mean((group["forward_confidence"] + group["reverse_confidence"]) / 2),
                        4,
                    ),
                }
            )
    pairwise_summary_df = pd.DataFrame(pairwise_summary_rows)

    merged = checklist_df.dropna(subset=["automatic_reward"])
    rho_default = stats.spearmanr(merged["automatic_reward"], merged["judge_overall_mean"])
    rho_checklist = stats.spearmanr(merged["automatic_reward"], merged["judge_checklist_mean"])

    out_dir = ROOT / "outputs" / "proxy_judge"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "checklist_judge_rows.csv", checklist_rows)
    checklist_summary.to_csv(out_dir / "checklist_judge_summary.csv", index=False, encoding="utf-8-sig")
    if not pairwise_df.empty:
        pairwise_df.to_csv(out_dir / "pairwise_judge_rows.csv", index=False, encoding="utf-8-sig")
    if not pairwise_summary_df.empty:
        pairwise_summary_df.to_csv(out_dir / "pairwise_judge_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {
            "spearman_reward_vs_judge_overall": round(float(rho_default.statistic), 4),
            "spearman_reward_vs_judge_overall_p": float(rho_default.pvalue),
            "spearman_reward_vs_judge_checklist": round(float(rho_checklist.statistic), 4),
            "spearman_reward_vs_judge_checklist_p": float(rho_checklist.pvalue),
        }
    ]).to_csv(out_dir / "judge_correlation.csv", index=False, encoding="utf-8-sig")

    print(f"Checklist judged rows: {len(checklist_rows)}")
    print(f"Pairwise judged rows: {len(pairwise_rows)}")
    print(f"Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
