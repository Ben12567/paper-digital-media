from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eai_co.core import (
    AUDIENCE_SEGMENTS,
    CampaignBriefEncoder,
    CreativeGenerator,
    EaiCoOptimizer,
    MultiObjectiveEvaluator,
    summarize_by_method,
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_products(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = load_json(ROOT / "configs" / "experiment_config.json")
    products = load_products(ROOT / "data" / "sample_products.csv")

    encoder = CampaignBriefEncoder()
    generator = CreativeGenerator(seed=42)
    main_evaluator = MultiObjectiveEvaluator(config["optimization"]["reward_weights"])
    optimizer = EaiCoOptimizer(
        generator=generator,
        evaluator=main_evaluator,
        rounds=config["optimization"]["rounds"],
        candidates_per_round=config["optimization"]["candidates_per_round"],
        elite_count=config["optimization"]["elite_count"],
    )

    final_candidates = []
    all_eai_candidates = []

    for row in products:
        for audience_name in config["audience_segments"]:
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
            final_candidates.extend(baseline_candidates)

            final, all_candidates = optimizer.optimize(brief)
            final_candidates.append(final)
            all_eai_candidates.extend(all_candidates)

            no_diversity_evaluator = MultiObjectiveEvaluator(
                config["optimization"]["reward_weights"],
                use_diversity=False,
            )
            no_diversity_optimizer = EaiCoOptimizer(
                generator=generator,
                evaluator=no_diversity_evaluator,
                rounds=config["optimization"]["rounds"],
                candidates_per_round=config["optimization"]["candidates_per_round"],
                elite_count=config["optimization"]["elite_count"],
            )
            final_no_diversity, candidates_no_diversity = no_diversity_optimizer.optimize(
                brief,
                method="Ours_without_diversity",
            )
            final_candidates.append(final_no_diversity)
            all_eai_candidates.extend(candidates_no_diversity)

            no_factual_evaluator = MultiObjectiveEvaluator(
                config["optimization"]["reward_weights"],
                use_factual_penalty=False,
            )
            no_factual_optimizer = EaiCoOptimizer(
                generator=generator,
                evaluator=no_factual_evaluator,
                rounds=config["optimization"]["rounds"],
                candidates_per_round=config["optimization"]["candidates_per_round"],
                elite_count=config["optimization"]["elite_count"],
            )
            final_no_factual, candidates_no_factual = no_factual_optimizer.optimize(
                brief,
                method="Ours_without_factual_penalty",
            )
            final_candidates.append(final_no_factual)
            all_eai_candidates.extend(candidates_no_factual)

            one_round_optimizer = EaiCoOptimizer(
                generator=generator,
                evaluator=main_evaluator,
                rounds=1,
                candidates_per_round=config["optimization"]["candidates_per_round"],
                elite_count=config["optimization"]["elite_count"],
            )
            final_one_round, candidates_one_round = one_round_optimizer.optimize(
                brief,
                method="Ours_without_iterative_loop",
            )
            final_candidates.append(final_one_round)
            all_eai_candidates.extend(candidates_one_round)

            general_brief = encoder.encode(
                row=row,
                audience=AUDIENCE_SEGMENTS[audience_name],
                platform=config["platform"],
                use_audience=False,
            )
            final_no_audience, candidates_no_audience = optimizer.optimize(
                general_brief,
                method="Ours_without_audience_modeling",
                evaluation_brief=brief,
            )
            final_candidates.append(final_no_audience)
            all_eai_candidates.extend(candidates_no_audience)

    candidate_rows = [candidate.to_dict() for candidate in final_candidates]
    summary_rows = summarize_by_method(final_candidates)

    write_jsonl(ROOT / "outputs" / "demo_candidates.jsonl", candidate_rows)
    write_summary_csv(ROOT / "outputs" / "demo_summary.csv", summary_rows)

    print("Demo complete")
    print(f"Final candidates: {len(final_candidates)}")
    print(f"EAI-CO explored candidates: {len(all_eai_candidates)}")
    print("Summary:")
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
