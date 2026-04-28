from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MODEL_PRICING = {
    "gpt-5.4-mini": {
        "input_per_million": 0.75,
        "output_per_million": 4.50,
    },
    "gpt-5.4": {
        "input_per_million": 2.50,
        "output_per_million": 15.00,
    },
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_model_name(model_name: str) -> str:
    if model_name in MODEL_PRICING:
        return model_name
    for candidate in MODEL_PRICING:
        if model_name.startswith(candidate + "-"):
            return candidate
    raise KeyError(model_name)


def primary_calls_per_task(rounds: int, candidates_per_round: int) -> int:
    baseline_calls = 1 + 1 + 1
    ours_calls = rounds * candidates_per_round
    return baseline_calls + ours_calls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "real_experiment_config_openai_mini_subset.json"))
    parser.add_argument("--input-tokens-per-call", type=int, default=420)
    parser.add_argument("--output-tokens-per-call", type=int, default=120)
    parser.add_argument("--high-input-tokens-per-call", type=int, default=650)
    parser.add_argument("--high-output-tokens-per-call", type=int, default=180)
    args = parser.parse_args()

    cfg = load_json(Path(args.config))
    model_name = cfg["text_provider"]["model"]
    pricing = MODEL_PRICING[normalize_model_name(model_name)]

    tasks = cfg["target_products"] * len(cfg["audience_segments"])
    calls_per_task = primary_calls_per_task(
        rounds=cfg["optimization"]["rounds"],
        candidates_per_round=cfg["optimization"]["candidates_per_round"],
    )
    total_calls = tasks * calls_per_task

    def estimate(input_tokens: int, output_tokens: int) -> float:
        total_input = total_calls * input_tokens
        total_output = total_calls * output_tokens
        return (
            total_input / 1_000_000 * pricing["input_per_million"]
            + total_output / 1_000_000 * pricing["output_per_million"]
        )

    base_cost = estimate(args.input_tokens_per_call, args.output_tokens_per_call)
    high_cost = estimate(args.high_input_tokens_per_call, args.high_output_tokens_per_call)

    print(f"model={model_name}")
    print(f"tasks={tasks}")
    print(f"calls_per_task={calls_per_task}")
    print(f"total_calls={total_calls}")
    print(f"base_assumption_input_tokens={args.input_tokens_per_call}")
    print(f"base_assumption_output_tokens={args.output_tokens_per_call}")
    print(f"estimated_base_cost_usd={base_cost:.4f}")
    print(f"high_assumption_input_tokens={args.high_input_tokens_per_call}")
    print(f"high_assumption_output_tokens={args.high_output_tokens_per_call}")
    print(f"estimated_high_cost_usd={high_cost:.4f}")


if __name__ == "__main__":
    main()
