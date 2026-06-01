from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eai_co.core import AUDIENCE_SEGMENTS, CampaignBriefEncoder, EaiCoOptimizer, MultiObjectiveEvaluator
from eai_co.real_providers import (
    DiffusersImageProvider,
    HuggingFaceInferenceImageProvider,
    HuggingFaceInferenceTextProvider,
    OpenAITextProvider,
    RealCreativeGenerator,
    TransformersTextProvider,
)
from eai_co.study import load_products, synthesize_products


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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric_int_fields = {"model_calls"}
    numeric_float_fields = {
        "latency_ms",
        "reward",
        "relevance",
        "clarity",
        "aesthetic",
        "audience_fit",
        "brand_safety",
        "factuality_penalty",
        "diversity",
        "predicted_engagement",
    }
    normalized: list[dict] = []
    for row in rows:
        item = dict(row)
        for key in numeric_int_fields:
            if key in item and item[key] not in ("", None):
                item[key] = int(float(item[key]))
        for key in numeric_float_fields:
            if key in item and item[key] not in ("", None):
                item[key] = float(item[key])
        normalized.append(item)
    return normalized


def candidate_row(candidate, task_id: str) -> dict:
    row = {
        "task_id": task_id,
        "candidate_id": candidate.candidate_id,
        "method": candidate.method,
        "product_id": candidate.product_id,
        "audience_segment": candidate.audience_segment,
        "headline": candidate.headline,
        "body": candidate.body,
        "cta": candidate.cta,
        "visual_prompt": candidate.visual_prompt,
        "model_calls": candidate.model_calls,
        "latency_ms": candidate.latency_ms,
    }
    row.update({f"axis_{k}": v for k, v in candidate.exploration_axes.items()})
    row.update(candidate.scores)
    return row


def summarize(rows: list[dict]) -> list[dict]:
    by_method: dict[str, list[dict]] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    summary_rows = []
    for method, items in sorted(by_method.items()):
        summary_rows.append(
            {
                "method": method,
                "n": len(items),
                "mean_reward": round(statistics.mean(item["reward"] for item in items), 4),
                "mean_relevance": round(statistics.mean(item["relevance"] for item in items), 4),
                "mean_clarity": round(statistics.mean(item["clarity"] for item in items), 4),
                "mean_audience_fit": round(statistics.mean(item["audience_fit"] for item in items), 4),
                "mean_diversity": round(statistics.mean(item["diversity"] for item in items), 4),
                "mean_predicted_engagement": round(statistics.mean(item["predicted_engagement"] for item in items), 4),
                "mean_model_calls": round(statistics.mean(item["model_calls"] for item in items), 2),
                "mean_latency_ms": round(statistics.mean(item["latency_ms"] for item in items), 2),
            }
        )
    return summary_rows


def expected_rows_per_task(primary_only: bool) -> int:
    return 5 if primary_only else 9


def load_resume_state(path: Path, primary_only: bool) -> tuple[list[dict], set[str]]:
    rows = load_csv_rows(path)
    if not rows:
        return [], set()
    counts = Counter(row["task_id"] for row in rows)
    expected = expected_rows_per_task(primary_only)
    completed = {task_id for task_id, count in counts.items() if count >= expected}
    filtered = [row for row in rows if row["task_id"] in completed]
    return filtered, completed


def build_text_provider(config: dict):
    text_cfg = config["text_provider"]
    kind = text_cfg["kind"]
    if kind == "openai":
        return OpenAITextProvider(model=text_cfg["model"])
    if kind == "transformers":
        return TransformersTextProvider(
            model_name_or_path=text_cfg["model_name_or_path"],
            revision=text_cfg.get("revision"),
        )
    if kind == "huggingface_inference":
        return HuggingFaceInferenceTextProvider(
            model=text_cfg["model"],
            provider=text_cfg.get("provider", "hf-inference"),
        )
    raise ValueError(f"Unsupported text provider: {kind}")


def build_image_provider(config: dict):
    image_cfg = config.get("image_provider", {})
    if not image_cfg.get("enabled"):
        return None
    kind = image_cfg["kind"]
    if kind == "diffusers":
        return DiffusersImageProvider(model_name_or_path=image_cfg["model_name_or_path"])
    if kind == "huggingface_inference":
        return HuggingFaceInferenceImageProvider(
            model=image_cfg["model"],
            provider=image_cfg.get("provider", "hf-inference"),
        )
    raise ValueError(f"Unsupported image provider: {kind}")


def build_optimizer(config: dict, generator, *, use_diversity: bool = True, use_factual_penalty: bool = True, rounds: int | None = None):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "local_qwen7b.json"))
    parser.add_argument("--limit-products", type=int, default=None)
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "real_generation"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_json(Path(args.config))
    base_products = load_products(ROOT / "data" / "sample_products.csv")
    products = synthesize_products(base_products, args.limit_products or config["target_products"])

    text_provider = build_text_provider(config)
    image_provider = build_image_provider(config)
    generator = RealCreativeGenerator(
        text_provider=text_provider,
        image_provider=image_provider,
        image_output_dir=Path(args.output_dir) / "images",
        seed=config.get("seed", 42),
    )
    encoder = CampaignBriefEncoder()
    baseline_evaluator = MultiObjectiveEvaluator(config["optimization"]["reward_weights"])
    main_optimizer = build_optimizer(config, generator)
    no_diversity_optimizer = build_optimizer(config, generator, use_diversity=False)
    no_factual_optimizer = build_optimizer(config, generator, use_factual_penalty=False)
    one_round_optimizer = build_optimizer(config, generator, rounds=1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / "real_candidates.csv"
    summary_path = out_dir / "real_summary.csv"

    if args.resume:
        rows, completed_tasks = load_resume_state(candidate_path, args.primary_only)
    else:
        rows, completed_tasks = [], set()

    task_counter = 0
    for product_row in products:
        for audience_name in config["audience_segments"]:
            task_counter += 1
            task_id = f"T{task_counter:04d}"
            if task_id in completed_tasks:
                print(f"Skipping completed task {task_id}")
                continue
            brief = encoder.encode(product_row, AUDIENCE_SEGMENTS[audience_name], config["platform"])

            task_rows: list[dict] = []
            baseline_candidates = [
                generator.generate_template(brief),
                generator.generate_single_shot(brief, "B1_SingleShot_API"),
                generator.generate_single_shot(brief, "B2_OpenSource_Only"),
                generator.generate_single_shot(brief, "B3_PromptEngineered_AI"),
            ]
            baseline_evaluator.evaluate_group(brief, baseline_candidates)
            task_rows.extend(candidate_row(candidate, task_id) for candidate in baseline_candidates)

            ours_final, _ = main_optimizer.optimize(brief, method="Ours_EAI_CO")
            task_rows.append(candidate_row(ours_final, task_id))

            if not args.primary_only:
                no_diversity_final, _ = no_diversity_optimizer.optimize(brief, method="Ours_without_diversity")
                task_rows.append(candidate_row(no_diversity_final, task_id))

                no_factual_final, _ = no_factual_optimizer.optimize(brief, method="Ours_without_factual_penalty")
                task_rows.append(candidate_row(no_factual_final, task_id))

                one_round_final, _ = one_round_optimizer.optimize(brief, method="Ours_without_iterative_loop")
                task_rows.append(candidate_row(one_round_final, task_id))

                general_brief = encoder.encode(
                    product_row,
                    AUDIENCE_SEGMENTS[audience_name],
                    config["platform"],
                    use_audience=False,
                )
                no_audience_final, _ = main_optimizer.optimize(
                    general_brief,
                    method="Ours_without_audience_modeling",
                    evaluation_brief=brief,
                )
                task_rows.append(candidate_row(no_audience_final, task_id))

            rows.extend(task_rows)
            completed_tasks.add(task_id)
            write_csv(candidate_path, rows)
            write_csv(summary_path, summarize(rows))
            print(f"Completed task {task_id}; cumulative rows={len(rows)}")

    write_csv(candidate_path, rows)
    write_csv(summary_path, summarize(rows))
    print(f"Generated {len(rows)} real-study candidates across {task_counter} tasks")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
