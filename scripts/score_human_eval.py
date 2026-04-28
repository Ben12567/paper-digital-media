from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


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
    return 0.0 if total == 0 else (greater - lower) / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--blind-key", required=True)
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "human_eval_scored"))
    args = parser.parse_args()

    items = pd.read_csv(args.items)
    ratings = pd.read_csv(args.ratings)
    blind_key = pd.read_csv(args.blind_key)
    merged = ratings.merge(items, on="item_id", how="inner").merge(blind_key, on="blind_method", how="left")
    merged["method"] = merged["actual_method"]

    summary = merged.groupby("method", as_index=False).agg(
        attractiveness=("attractiveness", "mean"),
        clarity=("clarity", "mean"),
        visual_quality=("visual_quality", "mean"),
        audience_relevance=("audience_relevance", "mean"),
        click_intention=("click_intention", "mean"),
    )

    pivot = merged.groupby(["rater_id", "method"], as_index=False)["click_intention"].mean()
    pivot = pivot.pivot(index="rater_id", columns="method", values="click_intention").dropna()
    methods = list(pivot.columns)
    friedman = stats.friedmanchisquare(*[pivot[m] for m in methods])

    rows = []
    if "Ours_EAI_CO" in methods:
        for method in methods:
            if method == "Ours_EAI_CO":
                continue
            stat = stats.wilcoxon(pivot["Ours_EAI_CO"], pivot[method], zero_method="wilcox", method="auto")
            rows.append(
                {
                    "baseline": method,
                    "ours_mean_click": round(float(pivot["Ours_EAI_CO"].mean()), 4),
                    "baseline_mean_click": round(float(pivot[method].mean()), 4),
                    "click_gain": round(float((pivot["Ours_EAI_CO"] - pivot[method]).mean()), 4),
                    "wilcoxon_statistic": float(stat.statistic),
                    "p_value": float(stat.pvalue),
                    "cliffs_delta": round(cliffs_delta(pivot["Ours_EAI_CO"].tolist(), pivot[method].tolist()), 4),
                }
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "human_method_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(out_dir / "human_significance_vs_ours.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [{
            "friedman_statistic": float(friedman.statistic),
            "friedman_p_value": float(friedman.pvalue),
        }]
    ).to_csv(out_dir / "human_global_statistics.csv", index=False, encoding="utf-8-sig")
    print(f"Scored human evaluation into {out_dir}")


if __name__ == "__main__":
    main()
