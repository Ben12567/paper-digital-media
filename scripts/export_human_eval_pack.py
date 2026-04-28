from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--sample-products", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    df = pd.read_csv(args.input)
    sampled_products = sorted(df["product_id"].unique())[: args.sample_products]
    sampled = df[df["product_id"].isin(sampled_products)].copy()

    blind_codes = {method: f"M{index + 1}" for index, method in enumerate(sorted(sampled["method"].unique()))}
    sampled["blind_method"] = sampled["method"].map(blind_codes)

    rating_rows = []
    for _, row in sampled.iterrows():
        rating_rows.append(
            {
                "item_id": row["candidate_id"],
                "task_id": row["task_id"],
                "product_id": row["product_id"],
                "audience_segment": row["audience_segment"],
                "blind_method": row["blind_method"],
                "headline": row["headline"],
                "body": row["body"],
                "cta": row["cta"],
                "visual_prompt": row["visual_prompt"],
                "generated_image_path": row.get("axis_generated_image_path", ""),
            }
        )

    pairwise_rows = []
    grouped = sampled.groupby("task_id")
    for task_id, group in grouped:
        if len(group) < 2:
            continue
        group = group.sample(frac=1.0, random_state=args.seed)
        rows = group.to_dict("records")
        if len(rows) >= 2:
            left, right = rows[0], rows[1]
            if rng.random() < 0.5:
                left, right = right, left
            pairwise_rows.append(
                {
                    "comparison_id": f"P_{task_id}",
                    "task_id": task_id,
                    "product_id": left["product_id"],
                    "audience_segment": left["audience_segment"],
                    "left_item_id": left["candidate_id"],
                    "left_blind_method": left["blind_method"],
                    "left_headline": left["headline"],
                    "left_body": left["body"],
                    "left_image_path": left.get("axis_generated_image_path", ""),
                    "right_item_id": right["candidate_id"],
                    "right_blind_method": right["blind_method"],
                    "right_headline": right["headline"],
                    "right_body": right["body"],
                    "right_image_path": right.get("axis_generated_image_path", ""),
                }
            )

    out_dir = ROOT / "outputs" / "human_eval_pack"
    write_csv(out_dir / "rating_items.csv", rating_rows)
    write_csv(out_dir / "pairwise_items.csv", pairwise_rows)
    (out_dir / "blind_key.csv").write_text(
        "blind_method,actual_method\n" + "\n".join(f"{v},{k}" for k, v in blind_codes.items()),
        encoding="utf-8-sig",
    )
    (out_dir / "rating_response_template.csv").write_text(
        "rater_id,item_id,attractiveness,clarity,visual_quality,audience_relevance,click_intention,comment\n",
        encoding="utf-8-sig",
    )
    (out_dir / "pairwise_response_template.csv").write_text(
        "rater_id,comparison_id,choice,reason\n",
        encoding="utf-8-sig",
    )
    print(f"Exported human evaluation pack to {out_dir}")


if __name__ == "__main__":
    main()
